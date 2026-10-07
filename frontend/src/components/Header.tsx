import React from "react";
import { ShieldCheck, Database, Cpu, Activity, BarChart3, Terminal } from "lucide-react";

interface HeaderProps {
  systemStatus: { status: string; services: Record<string, string> };
  devMode: boolean;
  onToggleDevMode: () => void;
  onOpenEvaluation: () => void;
  activeTenant: string;
  onTenantChange: (tenant: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemStatus,
  devMode,
  onToggleDevMode,
  onOpenEvaluation,
  activeTenant,
  onTenantChange,
}) => {
  const isHealthy = systemStatus.status === "healthy" || systemStatus.status === "ok";

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800 bg-[#090d16]/90 backdrop-blur-md px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3.5">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-700 via-indigo-600 to-cyan-500 shadow-glow">
            <ShieldCheck className="w-6 h-6 text-white" />
            <span className="absolute -bottom-1 -right-1 w-3 h-3 bg-emerald-500 border-2 border-[#090d16] rounded-full animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
                Sentinel<span className="text-blue-400">RAG</span>
              </h1>
              <span className="text-[10px] uppercase font-mono tracking-wider px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Enterprise Grounded
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Multi-Agent Document Intelligence & Evidence-Grounded Audit System
            </p>
          </div>
        </div>

        {/* Center / Stats & Status Indicators */}
        <div className="hidden md:flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800">
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" : "bg-amber-400"
              }`}
            />
            <span className="text-slate-300">
              {isHealthy ? "Cluster Online" : "Local Mode (Mock/Ready)"}
            </span>
          </div>

          <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-400">
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              Qdrant + BM25
            </span>
            <span className="text-slate-700">|</span>
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              LangGraph Multi-Agent
            </span>
          </div>
        </div>

        {/* Actions & Controls */}
        <div className="flex items-center gap-3">
          {/* Tenant Selector */}
          <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 font-mono">
            <span>Tenant:</span>
            <select
              value={activeTenant}
              onChange={(e) => onTenantChange(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-blue-500"
            >
              <option value="default_tenant">default_tenant</option>
              <option value="acme_corp">acme_corp</option>
              <option value="apex_finance">apex_finance</option>
            </select>
          </div>

          {/* Benchmark Evaluation Button */}
          <button
            onClick={onOpenEvaluation}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 rounded-lg transition-all"
            title="View RAG Evaluation Benchmark Results"
          >
            <BarChart3 className="w-3.5 h-3.5 text-blue-400" />
            <span className="hidden sm:inline">Benchmark</span>
          </button>

          {/* Developer Mode Toggle */}
          <button
            onClick={onToggleDevMode}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
              devMode
                ? "bg-blue-600/20 text-blue-300 border-blue-500/40 shadow-glow"
                : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>Dev Mode</span>
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                devMode ? "bg-blue-400 animate-ping" : "bg-slate-600"
              }`}
            />
          </button>
        </div>
      </div>
    </header>
  );
};
