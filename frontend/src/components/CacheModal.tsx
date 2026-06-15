import { useEffect, useState } from 'react';
import { api, CacheEntry } from '@/lib/api';

interface CacheModalProps {
  onClose: () => void;
}

function formatTTL(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h}h ${mm}m`;
}

export default function CacheModal({ onClose }: CacheModalProps) {
  const [entries, setEntries] = useState<CacheEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getCacheEntries()
      .then(res => {
        setEntries(res.entries);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load cache entries');
        setLoading(false);
      });
  }, []);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '800px', width: '90%' }}>
        <div className="modal-header">
          <h2 className="modal-title">Cached Responses</h2>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="modal-body" style={{ maxHeight: '60vh', overflowY: 'auto', padding: '0 4px' }}>
          {loading && <p style={{ color: 'var(--text-muted)' }}>Loading cache entries...</p>}
          {error && <p style={{ color: '#ef4444' }}>{error}</p>}
          
          {!loading && !error && entries.length === 0 && (
            <p style={{ color: 'var(--text-muted)' }}>No items currently in cache.</p>
          )}

          {!loading && !error && entries.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {entries.map((entry, idx) => (
                <div key={idx} style={{
                  background: 'var(--bg-lighter)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  padding: '12px',
                  fontSize: '13px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <strong style={{ color: 'var(--text)', wordBreak: 'break-word', flex: 1, marginRight: '16px' }}>
                      Q: {entry.question}
                    </strong>
                    <div style={{ display: 'flex', gap: '8px', flexShrink: 0, alignItems: 'flex-start' }}>
                      <span style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 600 }}>
                        TTL: {formatTTL(entry.ttl_seconds)}
                      </span>
                      {entry.hits > 0 && (
                        <span style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#6366f1', padding: '2px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 600 }}>
                          Hits: {entry.hits}
                        </span>
                      )}
                    </div>
                  </div>
                  <div style={{ color: 'var(--text-muted)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {entry.answer_preview}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
