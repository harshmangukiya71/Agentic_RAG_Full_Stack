'use client';

import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { EvalReport, EvalResult } from '@/lib/api';

interface EvalModalProps {
  report: EvalReport;
  onClose: () => void;
}

type Tab = 'retrieval' | 'generation' | 'latency';

const LATENCY_KEYS = [
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
  'reasoning_agent_time_ms',
  'generation_time_ms',
  'cache_lookup_time_ms',
  'total_response_time_ms',
];

const LABELS: Record<string, string> = {
  query_rewriter_time_ms: 'Query Rewriter Time',
  query_classifier_time_ms: 'Query Classifier Time',
  dense_retrieval_time_ms: 'Dense Retrieval Time',
  bm25_retrieval_time_ms: 'BM25 Retrieval Time',
  graph_retrieval_time_ms: 'Graph Retrieval Time',
  hybrid_retrieval_time_ms: 'Hybrid Retrieval Time',
  rrf_fusion_time_ms: 'RRF Fusion Time',
  evidence_validation_time_ms: 'Evidence Validation Time',
  graph_boosting_time_ms: 'Graph Boosting Time',
  cross_encoder_rerank_time_ms: 'Cross Encoder Rerank Time',
  reasoning_agent_time_ms: 'Reasoning Agent Time',
  generation_time_ms: 'Generation Time',
  cache_lookup_time_ms: 'Cache Lookup Time',
  total_response_time_ms: 'Total Response Time',
};

const TABS: { id: Tab; label: string }[] = [
  { id: 'retrieval', label: 'Retrieval Evaluation' },
  { id: 'generation', label: 'Generation Evaluation' },
  { id: 'latency', label: 'Latency Evaluation' },
];

export default function EvalModal({ report, onClose }: EvalModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>('retrieval');

  const pct = (value = 0) => `${(value * 100).toFixed(1)}%`;
  const score = (value = 0) => value.toFixed(3);
  const ms = (value = 0) => `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
  const yesNo = (value?: boolean) => (value ? 'Yes' : 'No');

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

  const latencySummary = useMemo(() => {
    const withTotals = report.results.map((result, index) => ({
      index,
      question: result.question,
      total: getLatencyValue(result, 'total_response_time_ms'),
    }));
    if (!withTotals.length) {
      return { slowest: undefined, fastest: undefined, p95: undefined };
    }
    const sortedTotals = withTotals.map(item => item.total).sort((a, b) => a - b);
    const slowest = withTotals.reduce((best, item) => (item.total > best.total ? item : best), withTotals[0]);
    const fastest = withTotals.reduce((best, item) => (item.total < best.total ? item : best), withTotals[0]);
    const p95Index = sortedTotals.length ? Math.ceil(sortedTotals.length * 0.95) - 1 : -1;

    return {
      slowest,
      fastest,
      p95: p95Index >= 0 ? sortedTotals[Math.min(p95Index, sortedTotals.length - 1)] : undefined,
    };
  }, [report.results]);

  const metricCard = (label: string, value: string, tone = '#7dd3fc') => (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid ${tone}44`,
      borderRadius: 8,
      padding: 12,
      minWidth: 0,
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 5 }}>{label}</div>
      <div style={{ fontSize: 21, fontWeight: 800, color: tone, lineHeight: 1.1 }}>{value}</div>
    </div>
  );

  const metricPill = (label: string, value: string, tone = '#7dd3fc') => (
    <div style={{
      border: `1px solid ${tone}44`,
      background: `${tone}12`,
      borderRadius: 8,
      padding: '8px 10px',
      minWidth: 108,
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>{label}</div>
      <strong style={{ color: tone, fontSize: 14 }}>{value}</strong>
    </div>
  );

  const sectionTitle = (title: string, subtitle?: string) => (
    <div>
      <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text)' }}>{title}</div>
      {subtitle && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>{subtitle}</div>}
    </div>
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        onClick={event => event.stopPropagation()}
        style={{ maxWidth: 1120, width: '96vw', maxHeight: '92vh', overflow: 'hidden' }}
      >
        <div className="modal-header">
          <div>
            <div className="modal-title">Evaluation Console</div>
            <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 3 }}>
              {report.total_questions} questions - {report.hits_at_3} successful retrievals @3 - MRR {score(report.mrr)}
            </div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">x</button>
        </div>

        <div style={{ display: 'flex', gap: 8, padding: '12px 20px 0', borderBottom: '1px solid var(--border)', flexWrap: 'wrap' }}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px 8px 0 0',
                border: 'none',
                cursor: 'pointer',
                fontWeight: activeTab === tab.id ? 800 : 600,
                background: activeTab === tab.id ? 'var(--accent)' : 'transparent',
                color: activeTab === tab.id ? '#fff' : 'var(--text-muted)',
                fontSize: 13,
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div style={{ padding: 20, maxHeight: 'calc(92vh - 124px)', overflowY: 'auto' }}>
          {activeTab === 'retrieval' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {sectionTitle('Per-question Retrieval', 'Inspect expected provenance, query rewrite behavior, ranks, and retrieval metrics for each question.')}
              {report.results.map((result, index) => (
                <QuestionCard key={`${result.question}-${index}`}>
                  <CardHeader
                    index={index}
                    question={result.question}
                    status={result.rank > 0 ? `Rank ${result.rank}` : 'Miss'}
                    tone={result.rank > 0 ? '#22c55e' : '#ef4444'}
                  />
                  <InfoGrid>
                    <Info label="Expected Document" value={result.expected_document} />
                    <Info label="Expected Page" value={`Page ${result.expected_page}`} />
                    <Info label="Original Query" value={result.original_query || result.question} />
                    <Info label="Rewritten Query" value={result.rewritten_query || 'Not rewritten'} />
                  </InfoGrid>
                  <MetricGrid>
                    {metricPill('Recall@1', pct(result.hit_at_1 ? 1 : 0), '#7dd3fc')}
                    {metricPill('Recall@3', pct(result.hit_at_3 ? 1 : 0), '#7dd3fc')}
                    {metricPill('Recall@5', pct(result.hit_at_5 ? 1 : 0), '#7dd3fc')}
                    {metricPill('Precision@1', pct(result.precision_at_1), '#6ee7b7')}
                    {metricPill('Precision@3', pct(result.precision_at_3), '#6ee7b7')}
                    {metricPill('Precision@5', pct(result.precision_at_5), '#6ee7b7')}
                    {metricPill('Hit@1', yesNo(result.hit_at_1), '#c4b5fd')}
                    {metricPill('Hit@3', yesNo(result.hit_at_3), '#c4b5fd')}
                    {metricPill('Hit@5', yesNo(result.hit_at_5), '#c4b5fd')}
                    {metricPill('Retrieved Rank', result.rank > 0 ? `#${result.rank}` : 'Not found', result.rank > 0 ? '#fbbf24' : '#ef4444')}
                    {metricPill('MRR Contribution', score(result.reciprocal_rank), '#fbbf24')}
                  </MetricGrid>
                </QuestionCard>
              ))}

              <AggregateBlock title="Overall Retrieval Metrics">
                {(['1', '3', '5'] as const).map(k => metricCard(`Recall@${k}`, pct(recall[k]), '#7dd3fc'))}
                {(['1', '3', '5'] as const).map(k => metricCard(`Precision@${k}`, pct(precision[k]), '#6ee7b7'))}
                {(['1', '3', '5'] as const).map(k => metricCard(`Hit@${k}`, pct(hit[k]), '#c4b5fd'))}
                {metricCard('MRR', score(retrieval.mrr ?? report.mrr), '#fbbf24')}
                {metricCard('Total Questions', String(report.total_questions), '#e5e7eb')}
                {metricCard('Successful Retrievals', String(report.hits_at_3), '#22c55e')}
              </AggregateBlock>
            </div>
          )}

          {activeTab === 'generation' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {sectionTitle('Per-question Generation', 'Compare generated and reference answers with answer quality metrics beside each sample.')}
              {report.results.map((result, index) => (
                <QuestionCard key={`${result.question}-${index}`}>
                  <CardHeader index={index} question={result.question} status={`BERT F1 ${pct(result.bertscore_f1)}`} tone="#f472b6" />
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
                    <AnswerPanel title="Generated Answer" text={result.generated_answer || 'No generated answer was returned for this evaluation result.'} />
                    <AnswerPanel title="Reference Answer" text={result.reference_answer || 'No reference answer was available for this question.'} />
                  </div>
                  <MetricGrid>
                    {metricPill('Faithfulness', pct(result.faithfulness), '#6ee7b7')}
                    {metricPill('Answer Relevancy', pct(result.answer_relevancy), '#7dd3fc')}
                    {metricPill('BERTScore Precision', pct(result.bertscore_precision), '#c4b5fd')}
                    {metricPill('BERTScore Recall', pct(result.bertscore_recall), '#fbbf24')}
                    {metricPill('BERTScore F1', pct(result.bertscore_f1), '#f472b6')}
                  </MetricGrid>
                </QuestionCard>
              ))}

              <AggregateBlock title="Overall Generation Metrics">
                {metricCard('Average Faithfulness', pct(generation.faithfulness), '#6ee7b7')}
                {metricCard('Average Answer Relevancy', pct(generation.answer_relevancy), '#7dd3fc')}
                {metricCard('Average BERTScore Precision', pct(generation.bertscore_precision), '#c4b5fd')}
                {metricCard('Average BERTScore Recall', pct(generation.bertscore_recall), '#fbbf24')}
                {metricCard('Average BERTScore F1', pct(generation.bertscore_f1), '#f472b6')}
              </AggregateBlock>
            </div>
          )}

          {activeTab === 'latency' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {sectionTitle('Per-question Latency', 'Break down each pipeline stage so bottlenecks are visible question by question.')}
              {report.results.map((result, index) => (
                <QuestionCard key={`${result.question}-${index}`}>
                  <CardHeader
                    index={index}
                    question={result.question}
                    status={ms(getLatencyValue(result, 'total_response_time_ms'))}
                    tone="#7dd3fc"
                  />
                  <LatencyTable result={result} ms={ms} />
                </QuestionCard>
              ))}

              <AggregateBlock title="Overall Latency Metrics">
                {LATENCY_KEYS.map(key => metricCard(`Average ${LABELS[key]}`, ms(latency[key] ?? 0), key === 'total_response_time_ms' ? '#fbbf24' : '#7dd3fc'))}
                {metricCard('Slowest Question', latencySummary.slowest ? `Q${latencySummary.slowest.index + 1} - ${ms(latencySummary.slowest.total)}` : 'n/a', '#ef4444')}
                {metricCard('Fastest Question', latencySummary.fastest ? `Q${latencySummary.fastest.index + 1} - ${ms(latencySummary.fastest.total)}` : 'n/a', '#22c55e')}
                {metricCard('P95 Total Latency', latencySummary.p95 !== undefined ? ms(latencySummary.p95) : 'n/a', '#f472b6')}
              </AggregateBlock>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function getLatencyValue(result: EvalResult, key: string) {
  if (key === 'total_response_time_ms') {
    return result.latency_metrics?.[key] ?? result.total_response_time_ms ?? 0;
  }
  return result.latency_metrics?.[key] ?? 0;
}

function QuestionCard({ children }: { children: ReactNode }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: 14,
      display: 'grid',
      gap: 12,
    }}>
      {children}
    </div>
  );
}

function CardHeader({ index, question, status, tone }: { index: number; question: string; status: string; tone: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Question {index + 1}</div>
        <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)', lineHeight: 1.35 }}>{question}</div>
      </div>
      <div style={{
        flexShrink: 0,
        border: `1px solid ${tone}55`,
        background: `${tone}18`,
        color: tone,
        borderRadius: 8,
        padding: '6px 9px',
        fontSize: 12,
        fontWeight: 800,
      }}>
        {status}
      </div>
    </div>
  );
}

function InfoGrid({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 }}>
      {children}
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 10, minWidth: 0 }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 12, color: 'var(--text)', overflowWrap: 'anywhere', lineHeight: 1.35 }}>{value}</div>
    </div>
  );
}

function MetricGrid({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(112px, 1fr))', gap: 8 }}>
      {children}
    </div>
  );
}

function AnswerPanel({ title, text }: { title: string; text: string }) {
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 12, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--text-muted)', marginBottom: 7 }}>{title}</div>
      <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.5, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
        {text}
      </div>
    </div>
  );
}

function LatencyTable({ result, ms }: { result: EvalResult; ms: (value?: number) => string }) {
  const maxValue = Math.max(...LATENCY_KEYS.map(key => getLatencyValue(result, key)), 1);

  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
      {LATENCY_KEYS.map(key => {
        const value = getLatencyValue(result, key);
        return (
          <div
            key={key}
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(160px, 1fr) minmax(120px, 2fr) 86px',
              gap: 10,
              alignItems: 'center',
              padding: '8px 10px',
              borderBottom: key === LATENCY_KEYS[LATENCY_KEYS.length - 1] ? 'none' : '1px solid var(--border)',
            }}
          >
            <div style={{ fontSize: 12, color: 'var(--text)' }}>{LABELS[key]}</div>
            <div style={{ height: 6, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
              <div style={{ width: `${Math.max(2, (value / maxValue) * 100)}%`, height: '100%', background: key === 'total_response_time_ms' ? '#fbbf24' : '#7dd3fc' }} />
            </div>
            <strong style={{ fontSize: 12, color: 'var(--text)', textAlign: 'right' }}>{ms(value)}</strong>
          </div>
        );
      })}
    </div>
  );
}

function AggregateBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{
      marginTop: 4,
      borderTop: '1px solid var(--border)',
      paddingTop: 16,
      display: 'grid',
      gap: 12,
    }}>
      <div style={{ fontSize: 15, fontWeight: 900, color: 'var(--text)' }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(155px, 1fr))', gap: 10 }}>
        {children}
      </div>
    </div>
  );
}
