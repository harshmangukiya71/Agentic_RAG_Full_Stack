'use client';

import { useState } from 'react';
import { EvalReport } from '@/lib/api';

interface EvalModalProps {
  report: EvalReport;
  onClose: () => void;
}

type Tab = 'retrieval' | 'generation' | 'time' | 'questions';

const LATENCY_KEYS = [
  'total_response_time_ms',
  'query_rewriter_time_ms',
  'query_classifier_time_ms',
  'dense_retrieval_time_ms',
  'bm25_retrieval_time_ms',
  'graph_retrieval_time_ms',
  'hybrid_retrieval_time_ms',
  'rrf_fusion_time_ms',
  'evidence_validation_time_ms',
  'graph_boosting_time_ms',
  'cross_encoder_rerank_time_ms',
  'reranking_time_ms',
  'reasoning_agent_time_ms',
  'generation_time_ms',
  'cache_lookup_time_ms',
];

const LABELS: Record<string, string> = {
  total_response_time_ms: 'Total response',
  query_rewriter_time_ms: 'Query rewriter',
  query_classifier_time_ms: 'Query classifier',
  dense_retrieval_time_ms: 'Dense retrieval',
  bm25_retrieval_time_ms: 'BM25 retrieval',
  graph_retrieval_time_ms: 'Graph retrieval',
  hybrid_retrieval_time_ms: 'Hybrid retrieval',
  rrf_fusion_time_ms: 'RRF fusion',
  evidence_validation_time_ms: 'Evidence validation',
  graph_boosting_time_ms: 'Graph boosting',
  cross_encoder_rerank_time_ms: 'Cross-encoder rerank',
  reranking_time_ms: 'Reranking',
  reasoning_agent_time_ms: 'Reasoning agent',
  generation_time_ms: 'Generation',
  cache_lookup_time_ms: 'Cache lookup',
};

export default function EvalModal({ report, onClose }: EvalModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>('retrieval');
  const pct = (value = 0) => `${(value * 100).toFixed(1)}%`;
  const ms = (value = 0) => `${value.toFixed(value >= 100 ? 0 : 1)} ms`;

  const retrieval = report.retrieval_metrics ?? {};
  const recall = retrieval.recall_at_k ?? {
    '1': report.recall_at_1,
    '3': report.recall_at_3,
    '5': report.recall_at_5,
  };
  const precision = retrieval.precision_at_k ?? {
    '1': report.precision_at_1 ?? 0,
    '3': report.precision_at_3,
    '5': report.precision_at_5 ?? 0,
  };
  const hit = retrieval.hit_rate ?? {
    '1': report.hit_rate_at_1 ?? report.recall_at_1,
    '3': report.hit_rate_at_3 ?? report.recall_at_3,
    '5': report.hit_rate_at_5 ?? report.recall_at_5,
  };
  const generation = report.generation_metrics ?? {};
  const latency = report.latency_metrics ?? {};
  const maxLatency = Math.max(...LATENCY_KEYS.map(key => latency[key] ?? 0), 1);

  const tabs: { id: Tab; label: string }[] = [
    { id: 'retrieval', label: 'Retrieval' },
    { id: 'generation', label: 'Generation' },
    { id: 'time', label: 'Time' },
    { id: 'questions', label: 'Per-question' },
  ];

  const metricCard = (label: string, value: string, accent: string) => (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid ${accent}33`,
      borderRadius: 8,
      padding: 14,
    }}>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color: accent }}>{value}</div>
    </div>
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        onClick={event => event.stopPropagation()}
        style={{ maxWidth: 860, width: '95vw' }}
      >
        <div className="modal-header">
          <div className="modal-title">Evaluation Report</div>
          <button className="modal-close" onClick={onClose} aria-label="Close">x</button>
        </div>

        <div style={{ textAlign: 'center', padding: '8px 0 4px', color: 'var(--text-muted)', fontSize: 13 }}>
          {report.total_questions} questions - {report.hits_at_3} hits @3 - MRR {report.mrr.toFixed(3)}
        </div>

        <div style={{ display: 'flex', gap: 8, padding: '12px 20px 0', borderBottom: '1px solid var(--border)', flexWrap: 'wrap' }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '7px 14px',
                borderRadius: '8px 8px 0 0',
                border: 'none',
                cursor: 'pointer',
                fontWeight: activeTab === tab.id ? 700 : 500,
                background: activeTab === tab.id ? 'var(--accent)' : 'transparent',
                color: activeTab === tab.id ? '#fff' : 'var(--text-muted)',
                fontSize: 13,
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === 'retrieval' && (
          <div style={{ padding: 20, display: 'grid', gap: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
              {(['1', '3', '5'] as const).map(k => metricCard(`Recall@${k}`, pct(recall[k]), '#7dd3fc'))}
              {(['1', '3', '5'] as const).map(k => metricCard(`Precision@${k}`, pct(precision[k]), '#6ee7b7'))}
              {(['1', '3', '5'] as const).map(k => metricCard(`Hit@${k}`, pct(hit[k]), '#c4b5fd'))}
              {metricCard('MRR', report.mrr.toFixed(3), '#fbbf24')}
            </div>
          </div>
        )}

        {activeTab === 'generation' && (
          <div style={{ padding: 20, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            {metricCard('Faithfulness', pct(generation.faithfulness), '#6ee7b7')}
            {metricCard('Answer relevancy', pct(generation.answer_relevancy), '#7dd3fc')}
            {metricCard('BERTScore precision', pct(generation.bertscore_precision), '#c4b5fd')}
            {metricCard('BERTScore recall', pct(generation.bertscore_recall), '#fbbf24')}
            {metricCard('BERTScore F1', pct(generation.bertscore_f1), '#f472b6')}
          </div>
        )}

        {activeTab === 'time' && (
          <div style={{ padding: 20, display: 'grid', gap: 10 }}>
            {LATENCY_KEYS.map(key => {
              const value = latency[key] ?? 0;
              return (
                <div key={key} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 12, marginBottom: 8 }}>
                    <span style={{ color: 'var(--text)' }}>{LABELS[key]}</span>
                    <strong>{ms(value)}</strong>
                  </div>
                  <div style={{ height: 5, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.max(2, (value / maxLatency) * 100)}%`, height: '100%', background: '#7dd3fc' }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === 'questions' && (
          <div style={{ padding: '16px 20px', maxHeight: '62vh', overflowY: 'auto' }}>
            {report.results.map((result, index) => {
              const rewritten = result.rewritten_query && result.original_query && result.rewritten_query !== result.original_query;
              return (
                <div
                  key={`${result.question}-${index}`}
                  style={{
                    marginBottom: 10,
                    padding: '12px 14px',
                    borderRadius: 8,
                    background: 'var(--surface)',
                    border: `1px solid ${result.hit_at_3 ? '#22c55e44' : '#ef444444'}`,
                    display: 'flex',
                    gap: 12,
                    alignItems: 'flex-start',
                  }}
                >
                  <div style={{
                    flexShrink: 0,
                    width: 44,
                    height: 36,
                    borderRadius: 8,
                    background: result.rank === 0 ? '#ef444422' : result.rank === 1 ? '#22c55e22' : '#fbbf2422',
                    color: result.rank === 0 ? '#ef4444' : result.rank === 1 ? '#22c55e' : '#fbbf24',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 800,
                    fontSize: result.rank === 0 ? 11 : 14,
                  }}>
                    {result.rank === 0 ? 'MISS' : `#${result.rank}`}
                  </div>

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 5 }}>
                      {result.question}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                      <span>Expected: <strong>{result.expected_document}</strong> pg.{result.expected_page}</span>
                      <span>Hit: <strong>{result.hit_at_5 ? 'yes' : 'no'}</strong></span>
                      <span>Faithfulness: <strong>{pct(result.faithfulness)}</strong></span>
                      <span>Relevancy: <strong>{pct(result.answer_relevancy)}</strong></span>
                      <span>BERT F1: <strong>{pct(result.bertscore_f1)}</strong></span>
                      <span>Total: <strong>{ms(result.total_response_time_ms)}</strong></span>
                    </div>
                    {rewritten && (
                      <div style={{ marginTop: 7, fontSize: 11, color: 'var(--text-muted)' }}>
                        {result.original_query} {'->'} <strong>{result.rewritten_query}</strong>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
