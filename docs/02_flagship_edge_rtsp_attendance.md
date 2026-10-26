# System Design RFC: Distributed Edge RTSP Multi-Camera Face Recognition & Vector Search Infrastructure

> **Classification:** Production System Design Whitepaper  
> **Author:** Machine Learning & Data Engineer (B.Sc. in Economics & Finance, BRAC University)  
> **Domain:** Edge Computing, Scalable Vector Retrieval & Workforce Labor Economics

---

## 1. Executive Summary

Enterprise workforce management across commercial facilities, factories, and corporate campuses requires reliable, real-time identity verification under continuous video streaming. Traditional badge card and biometric fingerprint systems suffer from physical bottlenecks, hardware wear, and widespread "buddy punching" (time-theft fraud where employees swipe on behalf of absent colleagues).

This whitepaper presents the architectural design of a distributed **Edge RTSP Multi-Camera Face Recognition & Vector Database System**. The platform ingests real-time H.264/H.265 RTSP streams across multiple camera endpoints, processes frames using INT8-quantized edge neural networks, performs approximate nearest neighbor (ANN) vector retrieval across **20,000+ enrolled employee identities** via PostgreSQL and `pgvector`, and enforces temporal deduplication state management.

```mermaid
flowchart LR
    A["RTSP Camera Streams\n(H.264/H.265 IP Feeds)"] --> B["Thread-Isolated\nStream Ingestion"]
    B --> C["YuNet INT8 Detection\n& 5-pt Keypoints"]
    C --> D["SFace INT8\n128-d Embedding"]
    D --> E["PostgreSQL + pgvector\n(IVFFlat ANN Search)"]
    E --> F["Temporal State Engine\n(Cooldown Deduplication)"]
    F --> G["Verified Attendance Event\n& Payroll Audit Log"]
```

---

## 2. Distributed Streaming & Edge Ingestion Tier

Continuous video streaming over RTSP (Real-Time Streaming Protocol) introduces network jitter, packet reordering, and socket buffer overflows if video capture and neural inference execute on the same execution thread.

```
RTSP Camera 1 ───> Thread 1 Worker (cv2.VideoCapture) ───> Circular Queue (Size=1) ───┐
RTSP Camera 2 ───> Thread 2 Worker (cv2.VideoCapture) ───> Circular Queue (Size=1) ───┼───> Multi-Stream Dispatcher
RTSP Camera N ───> Thread N Worker (cv2.VideoCapture) ───> Circular Queue (Size=1) ───┘
```

### 2.1 Thread-per-Camera Decoupling
To eliminate frame lag and internal buffer accumulation, each camera feed runs in an isolated OS worker thread:
* **The Frame Accumulation Trap:** If an RTSP stream produces 25 frames per second (fps) but inference consumes 50ms per frame (20 fps), a standard single-threaded loop accumulates a growing internal socket buffer, causing the live stream to display seconds or minutes of artificial latency.
* **The Circular Queue Solution:** The capture worker continuously reads frames from the socket and deposits only the latest frame into a single-element atomic queue (`collections.deque(maxlen=1)`). Older unconsumed frames are instantly dropped, ensuring the inference engine always processes the absolute newest live sensor frame.

### 2.2 Dynamic Frame Decimation
In physical attendance environments, human pedestrian transit velocity across a camera field of view ranges from 1.0 to 1.5 m/s. At 25 fps, a pedestrian remains visible for 40 to 100 consecutive frames. The engine applies configurable frame skipping ($k = 3 \text{ to } 5$), reducing required neural inference passes by 60%–80% without sacrificing detection coverage.

---

## 3. Quantized Edge Inference Tier

High-density deployments cannot afford dedicated server-grade GPUs for every camera cluster. The architecture is engineered around INT8-quantized edge models executing on ONNX Runtime with automatic hardware acceleration fallback:

$$\text{Execution Tier Priority: } \text{CUDA Execution Provider} \longrightarrow \text{CPU Execution Provider (AVX-512 / OpenVINO)}$$

| Pipeline Stage | Model Architecture | Quantization | Parameter Footprint | Output Representation |
| :--- | :--- | :---: | :---: | :--- |
| **Face Detection** | YuNet (OpenCV Zoo) | INT8 ONNX | ~250 KB | Bounding Box proposals + 5 landmarks |
| **Landmark Alignment** | Affine Similarity Transform | Native C++ | N/A | $112 \times 112$ canonical aligned crop |
| **Identity Embedding** | SFace (OpenCV / HF) | INT8 ONNX | ~1.5 MB | $128$-dimensional unit vector |
| **Anti-Spoofing** | MiniFASNetV2-SE | FP32 / FP16 | ~626 KB | Binary real vs spoof logit classification |

### 3.1 INT8 Quantization Mechanics
The models employ symmetric uniform 8-bit integer quantization, mapping 32-bit floating-point weights $W_f$ and activations $X_f$ to signed 8-bit integers:

$$q = \text{round}\left(\frac{x}{S}\right) + Z$$

Where $S$ is the scale factor and $Z = 0$ for symmetric quantization. This delivers a **$4\times$ memory reduction** and enables integer matrix multiplication (INT8 GEMM), yielding sub-10ms inference latencies even on low-cost edge CPU hardware.

---

## 4. Scalable Vector Retrieval Tier (PostgreSQL + `pgvector`)

Identifying a face from an enrolled population of 20,000+ employees requires fast similarity search over high-dimensional vector spaces.

### 4.1 Vector Storage Schema
Embeddings are stored natively using the `pgvector` extension:

```sql
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    employee_code VARCHAR(64) UNIQUE NOT NULL,
    full_name VARCHAR(128) NOT NULL,
    department VARCHAR(64),
    embedding vector(128) NOT NULL,
    enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 Inverted File Flat (IVFFlat) Indexing
Brute-force sequential scan ($k$-NN) over $N = 20,000$ vectors incurs an $O(N \cdot d)$ computational cost per frame, causing latency degradation as employee counts grow.

The system constructs an **Inverted File with Flat Quantization (IVFFlat)** index based on cosine distance:

```sql
CREATE INDEX IF NOT EXISTS employee_embedding_ivfflat_idx 
ON employees 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

#### Index Architecture:
1. **$k$-Means Clustering:** The vector space is partitioned into $C = 100$ Voronoi centroids during index creation.
2. **Sub-Space Querying:** At query time, the system compares the probe embedding against the 100 centroids, evaluating only the vectors residing in the nearest $n_{\text{probes}}$ clusters (e.g., $n_{\text{probes}} = 10$).
3. **Complexity Reduction:** This reduces candidate vector evaluations from 20,000 down to ~2,000, achieving sub-5ms query response times:

```sql
SELECT employee_code, full_name, (embedding <=> $1) AS cosine_distance 
FROM employees 
WHERE (embedding <=> $1) < 0.40 
ORDER BY embedding <=> $1 ASC 
LIMIT 1;
```

---

## 5. Temporal Deduplication State Machine

If an employee stands in front of a camera for 10 seconds, the inference engine recognizes them 50+ times. Without state management, this floods the payroll database with redundant entries.

```
Face Identified (ID: EMP_1042)
             │
             ▼
    Check In-Memory Cache
             │
    ┌────────┴────────┐
    │                 │
[Key Exists & Time < 300s]  [Key Absent OR Time >= 300s]
    │                 │
    ▼                 ▼
 Suppress Event    1. Write to Attendance DB Log
 (Discard)         2. Update Cache Key (TTL = 300s)
```

The platform maintains an in-memory key-value state store with sliding-window Time-To-Live (TTL):
* **State Key:** `attendance_state:{camera_id}:{employee_code}`
* **Deduplication Cooldown Window:** Configured to $\Delta t = 300\text{ seconds (5 minutes)}$.
* **Atomic Validation:** Any subsequent positive vector match within the active cooldown window is silently suppressed, guaranteeing exactly one valid check-in event per transit episode.

---

## 6. Applied Quantitative Financial & Labor Economics

### 6.1 The Principal-Agent Problem & Payroll Margin Protection
In labor economics, "buddy punching" represents an unobserved moral hazard enabled by information asymmetry:
* **The Economic Problem:** Hourly and salaried employees have an economic incentive to cover for absent or late peers, resulting in unearned wage payouts.
* **Empirical Cost:** According to the American Payroll Association (APA) and corporate labor audits, buddy punching accounts for **2.2% to 5.0% of total gross payroll costs** in shift-based and commercial enterprises.
* **Financial Formulation:** Let $W$ be total annual payroll. The annual deadweight economic loss $L_{\text{theft}}$ without biometric verification is:

$$L_{\text{theft}} = W \times \rho_{\text{theft}}, \quad \text{where } \rho_{\text{theft}} \in [0.02, 0.05]$$

For an enterprise with 1,000 employees earning an average annual salary of \$10,000:
* Annual Payroll: **\$10,000,000**.
* Recovered Payroll Margins: **\$220,000 to \$500,000 per year** through automated, non-repudiable biometric auditing.

### 6.2 Total Cost of Ownership (TCO): Edge CapEx vs. Cloud OpEx
A common architectural debate is whether to stream raw RTSP feeds to centralized cloud GPU clusters (AWS/GCP) or process on local edge devices.

Let's evaluate the Total Cost of Ownership ($TCO$) across a 3-year enterprise horizon for 20 camera streams:

| Cost Vector | Cloud Streaming Architecture (AWS EC2 GPU) | Local Edge Architecture (INT8 Quantized) |
| :--- | :--- | :--- |
| **Network Ingress/Egress** | 20 streams $\times$ 4 Mbps = 80 Mbps continuous bandwidth = **\$15,500 / yr** | Zero external video streaming bandwidth (internal LAN only) |
| **Compute Instances** | 24/7 `g4dn.xlarge` GPU instances = **\$18,200 / yr** | Standard on-prem edge micro-servers (One-time CapEx: **\$3,500**) |
| **Database Cost** | Managed AWS RDS with `pgvector` = **\$4,800 / yr** | Self-hosted PostgreSQL on internal network (Negligible) |
| **3-Year TCO** | **\$115,500 (Pure Recurring OpEx)** | **~\$8,500 (Initial Hardware CapEx + Power)** |
| **Net Financial Savings** | Baseline | **92.6% Cost Reduction (\$107,000 Saved)** |

By pairing lightweight INT8 neural networks with PostgreSQL `pgvector`, the architecture delivers enterprise-grade facial recognition at a fraction of standard cloud infrastructure expenditure.
