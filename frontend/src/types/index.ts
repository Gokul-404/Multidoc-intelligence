export interface DocumentItem {
  id: string;
  filename: string;
  pages: number;
  file_size_mb: number;
  language: string;
  document_type: string;
  is_scanned: boolean;
  total_chunks: number;
  status: "processing" | "ready" | "failed";
  uploaded_at: string;
  error?: string;
}

export interface Citation {
  citation_index: number;
  document_id: string;
  document_name: string;
  page_number: number;
  chunk_id: string;
  supporting_text: string;
}

export interface ClaimVerification {
  claim: string;
  supported: boolean;
  confidence: number;
  failure_reason?: string;
  supporting_chunk_ids: string[];
}

export interface AuditReport {
  status: "PASS" | "FAIL" | "WARN";
  confidence_score: number;
  grounded: boolean;
  claims_total: number;
  claims_supported: number;
  claims_unsupported: number;
  citations_valid: number;
  citations_invalid: number;
  has_numerical_claims: boolean;
  numerical_claims_verified: boolean;
  has_contradictions: boolean;
  contradiction_details?: string;
  failure_reasons: string[];
  claims_breakdown?: ClaimVerification[];
}

export interface PipelineStage {
  node: string;
  duration_ms: number;
  status: "success" | "retry" | "skipped";
  details?: Record<string, any>;
}

export interface DeveloperMetrics {
  original_query: string;
  rewritten_queries: string[];
  sub_questions: string[];
  retrieval_method: string;
  dense_results_count: number;
  bm25_results_count: number;
  fused_results_count: number;
  reranked_results_count: number;
  raw_tokens_count: number;
  compressed_tokens_count: number;
  compression_ratio: number;
  pipeline_stages: PipelineStage[];
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
  retrieved_chunks?: Array<{
    chunk_id: string;
    document_name: string;
    page: number;
    score: number;
    retrieval_method: string;
    text: string;
  }>;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_score: number;
  grounded: boolean;
  refused: boolean;
  refusal_reason?: string;
  retries_attempted: number;
  audit: AuditReport;
  developer_mode?: DeveloperMetrics;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  confidence_score?: number;
  grounded?: boolean;
  refused?: boolean;
  refusal_reason?: string;
  audit?: AuditReport;
  developer_mode?: DeveloperMetrics;
}
