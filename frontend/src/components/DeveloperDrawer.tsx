import React, { useState } from "react";
import { DeveloperMetrics, PipelineStage } from "../types";
import {
  X,
  Terminal,
  Cpu,
  GitBranch,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  FileCheck,
  Zap,
  Activity,
  Maximize2,
} from "lucide-react";

interface DeveloperDrawerProps {
  metrics: DeveloperMetrics | null;
  isOpen: boolean;
  onClose: () => void;
}

export const DeveloperDrawer: React.FC<DeveloperDrawerProps> = ({
  metrics,
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<
    "pipeline" | "rewriting" | "retrieval" | "compression" | "chunks"
  >("pipeline");

  if (!isOpen || !metrics) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[600px] lg:w-[720px] bg-[#0b101d] border-l border-slate-800 shadow-2xl flex flex-col backdrop-blur-xl animate-in slide-in-from-right duration-200">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0f172a]/70">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              LangGraph Execution & Evidence Inspector
            </h3>
            <p className="text-[11px] font-mono text-slate-400">
              Trace latency: {metrics.total_latency_ms?.toFixed(0)}ms • RRF Hybrid Search
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 px-6 border-b border-slate-800 bg-slate-900/40 text-xs font-mono overflow-x-auto">
        <button
          onClick={() => setActiveTab("pipeline")}
          className={`px-3 py-3 border-b-2 font-medium transition-all ${
            activeTab === "pipeline"
              ? "border-blue-500 text-blue-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          Pipeline Trace
        </button>
        <button
          onClick={() => setActiveTab("rewriting")}
          className={`px-3 py-3 border-b-2 font-medium transition-all ${
            activeTab === "rewriting"
              ? "border-blue-500 text-blue-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          Query Rewriting
        </button>
        <button
          onClick={() => setActiveTab("retrieval")}
          className={`px-3 py-3 border-b-2 font-medium transition-all ${
            activeTab === "retrieval"
              ? "border-blue-500 text-blue-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          Hybrid RRF & Rerank
        </button>
        <button
          onClick={() => setActiveTab("compression")}
          className={`px-3 py-3 border-b-2 font-medium transition-all ${
            activeTab === "compression"
              ? "border-blue-500 text-blue-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          Compression & Tokens
        </button>
        <button
          onClick={() => setActiveTab("chunks")}
          className={`px-3 py-3 border-b-2 font-medium transition-all ${
            activeTab === "chunks"
              ? "border-blue-500 text-blue-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          Retrieved Chunks
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* TAB 1: PIPELINE TRACE */}
        {activeTab === "pipeline" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-200 mb-2 font-mono flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5 text-blue-400" />
                LangGraph StateGraph Nodes Traversed
              </h4>
              <div className="space-y-2.5 mt-3">
                {metrics.pipeline_stages?.map((stage, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 text-xs"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="w-5 h-5 rounded-full bg-blue-500/10 text-blue-400 flex items-center justify-center font-mono text-[10px]">
                        {idx + 1}
                      </span>
                      <span className="font-mono text-slate-300">{stage.node}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-[11px] text-slate-400">
                        {stage.duration_ms}ms
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {stage.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance Overview */}
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">Retrieval</span>
                <p className="text-sm font-bold text-white font-mono mt-1">
                  {metrics.retrieval_latency_ms?.toFixed(0)}ms
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">Generation</span>
                <p className="text-sm font-bold text-white font-mono mt-1">
                  {metrics.generation_latency_ms?.toFixed(0)}ms
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">Total</span>
                <p className="text-sm font-bold text-cyan-400 font-mono mt-1">
                  {metrics.total_latency_ms?.toFixed(0)}ms
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: QUERY REWRITING */}
        {activeTab === "rewriting" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h4 className="text-xs font-semibold text-slate-300 font-mono">
                Original User Query
              </h4>
              <p className="text-xs text-slate-200 bg-slate-950 p-3 rounded-lg border border-slate-800">
                {metrics.original_query}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h4 className="text-xs font-semibold text-slate-300 font-mono">
                Rewritten Query Variants (Multi-Query Retrieval)
              </h4>
              <ul className="space-y-2">
                {metrics.rewritten_queries?.map((rq, idx) => (
                  <li
                    key={idx}
                    className="p-2.5 rounded-lg bg-slate-950 text-xs text-slate-300 border border-slate-800 flex items-start gap-2"
                  >
                    <ArrowRight className="w-3.5 h-3.5 text-blue-400 shrink-0 mt-0.5" />
                    <span>{rq}</span>
                  </li>
                ))}
              </ul>
            </div>

            {metrics.sub_questions && metrics.sub_questions.length > 0 && (
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                <h4 className="text-xs font-semibold text-slate-300 font-mono">
                  Decomposed Sub-Questions (Multi-Hop)
                </h4>
                <ul className="space-y-2">
                  {metrics.sub_questions.map((sq, idx) => (
                    <li
                      key={idx}
                      className="p-2.5 rounded-lg bg-slate-950 text-xs text-slate-300 border border-slate-800 flex items-start gap-2"
                    >
                      <span className="text-blue-400 font-mono text-[11px]">#{idx + 1}</span>
                      <span>{sq}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: HYBRID RRF & RERANK */}
        {activeTab === "retrieval" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">Dense Hits</span>
                <p className="text-sm font-bold text-white font-mono mt-1">
                  {metrics.dense_results_count}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">BM25 Hits</span>
                <p className="text-sm font-bold text-white font-mono mt-1">
                  {metrics.bm25_results_count}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">RRF Fused</span>
                <p className="text-sm font-bold text-blue-400 font-mono mt-1">
                  {metrics.fused_results_count}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-mono text-slate-400">Reranked</span>
                <p className="text-sm font-bold text-emerald-400 font-mono mt-1">
                  {metrics.reranked_results_count}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-300 font-mono mb-2">
                Reciprocal Rank Fusion (RRF) Formula
              </h4>
              <p className="text-xs text-slate-400 font-mono bg-slate-950 p-2.5 rounded border border-slate-800">
                RRF_score(d) = Σ [ weight / (60 + rank_i(d)) ]
              </p>
              <p className="text-[11px] text-slate-500 mt-2">
                Dense (Qdrant cosine embeddings) + Sparse (BM25 token inverted index)
                deduplicated and reranked using local FlashRank cross-encoder.
              </p>
            </div>
          </div>
        )}

        {/* TAB 4: COMPRESSION & TOKENS */}
        {activeTab === "compression" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h4 className="text-xs font-semibold text-slate-300 font-mono">
                Contextual Compression Ratio
              </h4>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Parent Chunks Raw Tokens:</span>
                <span className="text-slate-200 font-bold">{metrics.raw_tokens_count} tokens</span>
              </div>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Compressed Context Tokens:</span>
                <span className="text-emerald-400 font-bold">
                  {metrics.compressed_tokens_count} tokens
                </span>
              </div>
              <div className="flex items-center justify-between text-xs font-mono border-t border-slate-800 pt-2">
                <span className="text-slate-400">Token Reduction Ratio:</span>
                <span className="text-blue-400 font-bold">
                  {(metrics.compression_ratio * 100).toFixed(1)}% savings
                </span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: RETRIEVED CHUNKS */}
        {activeTab === "chunks" && (
          <div className="space-y-3">
            {metrics.retrieved_chunks?.map((chunk, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 text-xs"
              >
                <div className="flex items-center justify-between font-mono text-[11px]">
                  <span className="text-blue-400 font-semibold">{chunk.document_name}</span>
                  <span className="text-slate-400">
                    Page {chunk.page} • Score: {chunk.score.toFixed(3)}
                  </span>
                </div>
                <p className="text-slate-300 font-mono text-[11px] leading-relaxed bg-slate-950 p-2.5 rounded border border-slate-800/80">
                  {chunk.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
