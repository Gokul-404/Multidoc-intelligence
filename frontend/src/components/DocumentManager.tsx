import React, { useState, useRef } from "react";
import { DocumentItem } from "../types";
import {
  FileText,
  UploadCloud,
  Trash2,
  CheckCircle2,
  AlertCircle,
  FileSearch,
  Filter,
  CheckSquare,
  Square,
  Layers,
  Sparkles,
} from "lucide-react";

interface DocumentManagerProps {
  documents: DocumentItem[];
  selectedDocIds: string[];
  onToggleDocSelect: (docId: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onUpload: (file: File) => Promise<void>;
  onDelete: (docId: string) => Promise<void>;
  isUploading: boolean;
}

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  documents,
  selectedDocIds,
  onToggleDocSelect,
  onSelectAll,
  onClearSelection,
  onUpload,
  onDelete,
  isUploading,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const filteredDocs = documents.filter((d) =>
    d.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith(".pdf")) {
        await onUpload(file);
      } else {
        alert("Please upload a PDF document.");
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await onUpload(e.target.files[0]);
      e.target.value = "";
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0c121e]/80 border-r border-slate-800/80 w-80 lg:w-96 p-4">
      {/* Title & Stats */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold tracking-wide text-slate-200">
            Document Repository
          </h2>
        </div>
        <span className="text-xs font-mono text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
          {documents.length} Files
        </span>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center p-5 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
          isDragOver
            ? "border-blue-500 bg-blue-500/10 scale-[1.01]"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/50 hover:bg-slate-900/80"
        } ${isUploading ? "opacity-60 pointer-events-none" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center mb-2.5 text-blue-400">
          <UploadCloud className="w-5 h-5" />
        </div>
        <p className="text-xs font-medium text-slate-300 text-center mb-1">
          {isUploading ? "Ingesting & Chunking PDF..." : "Upload Any PDF Document"}
        </p>
        <p className="text-[11px] text-slate-500 text-center">
          Contracts, Reports, Research, Invoices (OCR supported)
        </p>
        {isUploading && (
          <div className="w-full mt-3 bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div className="bg-blue-500 h-1.5 rounded-full animate-pulse w-3/4" />
          </div>
        )}
      </div>

      {/* Search & Bulk Select Controls */}
      <div className="mt-4 mb-3 space-y-2">
        <div className="relative">
          <FileSearch className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-slate-900/90 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
          <span>
            {selectedDocIds.length === 0
              ? "All documents active (global)"
              : `${selectedDocIds.length} of ${documents.length} active`}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onSelectAll}
              className="hover:text-blue-400 transition-colors"
            >
              Select All
            </button>
            <span className="text-slate-700">|</span>
            <button
              onClick={onClearSelection}
              className="hover:text-slate-300 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {filteredDocs.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-500">
            No matching documents found.
          </div>
        ) : (
          filteredDocs.map((doc) => {
            const isSelected = selectedDocIds.includes(doc.id);
            return (
              <div
                key={doc.id}
                onClick={() => onToggleDocSelect(doc.id)}
                className={`p-3 rounded-xl border transition-all cursor-pointer group ${
                  isSelected
                    ? "bg-blue-950/30 border-blue-500/40 shadow-sm"
                    : "bg-slate-900/40 border-slate-800/80 hover:bg-slate-900/80 hover:border-slate-700"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2.5 overflow-hidden">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleDocSelect(doc.id);
                      }}
                      className="mt-0.5 text-slate-400 hover:text-blue-400 transition-colors"
                    >
                      {isSelected ? (
                        <CheckSquare className="w-3.5 h-3.5 text-blue-400" />
                      ) : (
                        <Square className="w-3.5 h-3.5 text-slate-600" />
                      )}
                    </button>
                    <div className="overflow-hidden">
                      <p className="text-xs font-medium text-slate-200 truncate group-hover:text-white">
                        {doc.filename}
                      </p>
                      <div className="flex items-center gap-2 mt-1.5 flex-wrap text-[10px] font-mono text-slate-400">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/50">
                          {doc.pages ?? 1} {(doc.pages ?? 1) === 1 ? "page" : "pages"}
                        </span>
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/50">
                          {doc.total_chunks ?? 0} chunks
                        </span>
                        {doc.is_scanned && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                            OCR
                          </span>
                        )}
                        <span className="text-slate-500">
                          {(typeof doc.file_size_mb === "number" ? doc.file_size_mb : 0).toFixed(1)} MB
                        </span>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete ${doc.filename}?`)) {
                        onDelete(doc.id);
                      }
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-all rounded hover:bg-slate-800"
                    title="Delete document"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer / Isolation Badge */}
      <div className="pt-3 mt-2 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          Tenant Isolation Guaranteed
        </span>
        <span className="font-mono text-slate-400">Qdrant dense + BM25</span>
      </div>
    </div>
  );
};
