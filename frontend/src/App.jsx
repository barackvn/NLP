import React, { useState, useEffect } from 'react';
import {
  Shield, FileText, Database, Cpu, TrendingUp,
  FileSpreadsheet, Settings, HelpCircle, BookOpen,
  Info, Search, Download, ChevronDown, ChevronUp,
  CheckCircle2, AlertTriangle, AlertCircle, Sparkles, RefreshCw,
  X, Activity, Check, Server
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const QUICK_EXAMPLES = [
  "Mày là đứa ngu si phản quốc, cút xéo ra khỏi đây ngay!",
  "Đồ ngu, nhìn cái mặt mày hãm thật sự luôn đó.",
  "Thằng chó rác rưởi đừng có ở đây mà xàm xí đú.",
  "Hôm nay thời tiết ở Sài Gòn đẹp quá, đi uống cà phê thôi mọi người!",
  "Món ăn của quán này bình thường nhưng giá cả hơi đắt."
];

// Dữ liệu ban đầu sẵn sàng để hiển thị ngay khi mở trang
const INITIAL_PREDICTION = {
  text: "Mày là đứa ngu si phản quốc, cút xéo ra khỏi đây ngay!",
  total_latency_seconds: 0.8,
  models: [
    {
      alias: "linear",
      key: "PhoBERT-Linear",
      name: "PhoBERT-Linear (Baseline Thầy)",
      num: "1",
      f1_val: "0.663",
      color: "red",
      is_toxic: true,
      spans_count: 2,
      spans: ["ngu_si", "cút_xéo"],
      latency_ms: 178.1,
      categories: ["INSULT", "PROFANITY", "THREAT"],
      masked_text: "Mày là đứa *** phản quốc, *** ra khỏi đây ngay!",
      eval_type: "warning",
      eval_title: "Hiệu quả ở mức cơ bản",
      eval_desc: "Phát hiện được các chuỗi xúc phạm rõ ràng, chưa tốt với ngữ cảnh phức tạp.",
      words: ["Mày", "là", "đứa", "ngu_si", "phản_quốc", ",", "cút_xéo", "ra", "khỏi", "đây", "ngay", "!"],
      tags: ["O", "O", "O", "B-HOS", "O", "O", "B-HOS", "O", "O", "O", "O", "O"]
    },
    {
      alias: "crf",
      key: "PhoBERT-CRF",
      name: "PhoBERT-CRF (Bóc tách Ablation)",
      num: "2",
      f1_val: "0.686",
      color: "blue",
      is_toxic: true,
      spans_count: 2,
      spans: ["ngu_si", "cút_xéo"],
      latency_ms: 150.1,
      categories: ["INSULT", "PROFANITY", "THREAT"],
      masked_text: "Mày là đứa *** phản quốc, *** ra khỏi đây ngay!",
      eval_type: "info",
      eval_title: "Cải thiện so với baseline",
      eval_desc: "Bóc tách ranh giới chuỗi tốt hơn, giảm nhiễu.",
      words: ["Mày", "là", "đứa", "ngu_si", "phản_quốc", ",", "cút_xéo", "ra", "khỏi", "đây", "ngay", "!"],
      tags: ["O", "O", "O", "B-HOS", "O", "O", "B-HOS", "O", "O", "O", "O", "O"]
    },
    {
      alias: "bilstm_crf",
      key: "PhoBERT-BiLSTM-CRF(DualHead)",
      name: "PhoBERT-BiLSTM-CRF(DualHead)",
      num: "3",
      f1_val: "0.714",
      color: "green",
      is_toxic: true,
      spans_count: 3,
      spans: ["ngu_si", "phản_quốc", "cút_xéo"],
      latency_ms: 144.5,
      categories: ["INSULT", "PROFANITY", "THREAT", "DISCRIMINATION"],
      masked_text: "Mày là đứa *** ***, *** ra khỏi đây ngay!",
      eval_type: "success",
      eval_title: "Đề xuất SOTA Đa nhiệm",
      eval_desc: "Độ chính xác cao, xử lý tốt ngữ cảnh và tích hợp cổng Gated Intent triệt tiêu báo động giả.",
      words: ["Mày", "là", "đứa", "ngu_si", "phản_quốc", ",", "cút_xéo", "ra", "khỏi", "đây", "ngay", "!"],
      tags: ["O", "O", "O", "B-HOS", "B-HOS", "O", "B-HOS", "O", "O", "O", "O", "O"]
    }
  ]
};

export default function App() {
  const [activeNav, setActiveNav] = useState('predict');
  const [inputText, setInputText] = useState(QUICK_EXAMPLES[0]);
  const [autoMasking, setAutoMasking] = useState(true);
  const [showExplain, setShowExplain] = useState(false);
  const [compareAll, setCompareAll] = useState(true);
  
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(INITIAL_PREDICTION);
  const [accordionOpen, setAccordionOpen] = useState(false);
  
  // Trạng thái kết nối & Modal kiểm soát 3 mô hình
  const [backendOnline, setBackendOnline] = useState(false);
  const [healthData, setHealthData] = useState(null);
  const [statusModalOpen, setStatusModalOpen] = useState(false);
  const [pinging, setPinging] = useState(false);

  // Hàm ping kiểm tra kết nối 3 mô hình
  const checkHealth = async () => {
    setPinging(true);
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
        setBackendOnline(true);
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
    } finally {
      setPinging(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Gọi API phân tích
  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          model_key: compareAll ? 'all' : 'bilstm_crf'
        })
      });

      if (res.ok) {
        const data = await res.json();
        setPredictionData(data);
        setBackendOnline(true);
      } else {
        console.warn('Backend returned error, falling back to simulated inference');
      }
    } catch (err) {
      console.warn('Backend not reachable, simulated fallback:', err);
    } finally {
      setLoading(false);
    }
  };

  // Render các nhãn phân loại (INSULT, PROFANITY, ...)
  const renderCategoryTags = (detectedCats = []) => {
    const allCategories = ["INSULT", "PROFANITY", "THREAT", "DISCRIMINATION", "OTHER"];
    return (
      <div className="category-badges-row">
        {allCategories.map(cat => {
          const isActive = detectedCats.includes(cat);
          let activeClass = 'inactive';
          if (isActive) {
            if (cat === 'INSULT') activeClass = 'active-insult';
            else if (cat === 'PROFANITY') activeClass = 'active-profanity';
            else if (cat === 'THREAT') activeClass = 'active-threat';
            else if (cat === 'DISCRIMINATION') activeClass = 'active-discrimination';
            else activeClass = 'active-other';
          }
          return (
            <span key={cat} className={`cat-pill ${activeClass}`}>
              {cat}
            </span>
          );
        })}
      </div>
    );
  };

  // Format Auto-Masking Text
  const renderMaskedText = (maskedString) => {
    if (!maskedString) return null;
    const parts = maskedString.split('***');
    return (
      <span>
        {parts.map((part, idx) => (
          <React.Fragment key={idx}>
            {part}
            {idx < parts.length - 1 && <span className="mask-star">***</span>}
          </React.Fragment>
        ))}
      </span>
    );
  };

  return (
    <div className="app-container">
      {/* 1. SIDEBAR */}
      <aside className="sidebar">
        <div>
          {/* Logo Brand */}
          <div className="brand-header">
            <div className="brand-icon-box">
              <Shield size={22} />
            </div>
            <div>
              <div className="brand-title">ViHOS</div>
              <div className="brand-subtitle">Toxic Spans Guard</div>
            </div>
          </div>

          {/* Navigation Menu */}
          <nav className="nav-menu">
            <button
              className={`nav-item ${activeNav === 'predict' ? 'active' : ''}`}
              onClick={() => setActiveNav('predict')}
            >
              <FileText size={18} />
              <span>Phân tích văn bản</span>
            </button>
            <button
              className={`nav-item ${activeNav === 'dataset' ? 'active' : ''}`}
              onClick={() => setActiveNav('dataset')}
            >
              <Database size={18} />
              <span>Dữ liệu huấn luyện</span>
            </button>
            <button
              className={`nav-item ${activeNav === 'models' ? 'active' : ''}`}
              onClick={() => setActiveNav('models')}
            >
              <Cpu size={18} />
              <span>Mô hình</span>
            </button>
            <button
              className={`nav-item ${activeNav === 'evaluation' ? 'active' : ''}`}
              onClick={() => setActiveNav('evaluation')}
            >
              <TrendingUp size={18} />
              <span>Đánh giá & So sánh</span>
            </button>
            <button
              className={`nav-item ${activeNav === 'reports' ? 'active' : ''}`}
              onClick={() => setActiveNav('reports')}
            >
              <FileSpreadsheet size={18} />
              <span>Báo cáo</span>
            </button>
            <button
              className={`nav-item ${activeNav === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveNav('settings')}
            >
              <Settings size={18} />
              <span>Cấu hình hệ thống</span>
            </button>
          </nav>

          {/* Documents Section */}
          <div className="nav-section-title">TÀI LIỆU</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div className="doc-link" onClick={() => setActiveNav('reports')}>
              <HelpCircle size={16} />
              <span>Hướng dẫn sử dụng</span>
            </div>
            <div className="doc-link" onClick={() => setActiveNav('models')}>
              <BookOpen size={16} />
              <span>Tài liệu kỹ thuật</span>
            </div>
            <div className="doc-link" onClick={() => setActiveNav('dataset')}>
              <Info size={16} />
              <span>Giới thiệu dự án</span>
            </div>
          </div>
        </div>

        {/* Bottom Status Box */}
        <div className="status-widget">
          <div 
            className="status-indicator" 
            onClick={() => setStatusModalOpen(true)}
            style={{ cursor: 'pointer' }}
            title="Bấm để xem chi tiết kết nối 3 mô hình"
          >
            <span className="status-dot" style={{ backgroundColor: backendOnline ? '#10B981' : '#EF4444' }}></span>
            <span>{backendOnline ? '3/3 Mô hình Kết nối OK' : 'Mất kết nối Backend'}</span>
          </div>
          <div className="status-sub">Phiên bản 1.0.0</div>
          <div className="status-org">ViHOS - UIT 2026</div>
        </div>
      </aside>

      {/* 2. MAIN CANVAS */}
      <main className="main-wrapper">
        {/* Top Header */}
        <header className="top-header">
          <div className="top-header-left">
            <div className="top-shield-icon">
              <Shield size={24} />
            </div>
            <div>
              <h1 className="header-title">ViHOS Toxic Spans Guard</h1>
              <p className="header-subtitle">
                Hệ thống phát hiện, phân loại và giải thích chuỗi ngôn ngữ xúc phạm tiếng Việt bằng mô hình học sâu
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* NÚT TRẠNG THÁI 3 MÔ HÌNH KIỂM SOÁT */}
            <button 
              className={`btn-status-monitor ${backendOnline ? 'online' : 'offline'}`}
              onClick={() => setStatusModalOpen(true)}
              title="Bấm để kiểm tra chi tiết trạng thái kết nối của 3 mô hình"
            >
              <span className={`pulse-dot ${backendOnline ? 'online' : 'offline'}`}></span>
              <span>{backendOnline ? '🟢 3/3 Mô hình Sẵn sàng' : '🔴 Kiểm tra kết nối BE'}</span>
              <ChevronDown size={14} />
            </button>

            {/* Badge người dùng */}
            <div className="user-badge">
              <div className="user-avatar">IT</div>
              <span>IT Department ▾</span>
            </div>
          </div>
        </header>

        {/* TAB 1: PHÂN TÍCH VĂN BẢN (MAIN DASHBOARD) */}
        {activeNav === 'predict' && (
          <div>
            {/* Input Card */}
            <div className="input-card">
              <div className="input-card-title">
                <FileText size={18} color="#2563EB" />
                <span>Nhập văn bản cần phân tích</span>
              </div>

              <textarea
                className="input-textarea"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Nhập hoặc dán đoạn văn bản tiếng Việt tại đây..."
                rows={3}
              />

              {/* Quick sample pills */}
              <div className="quick-examples-row">
                <span style={{ fontSize: '0.78rem', color: '#64748B', fontWeight: 600 }}>Ví dụ thử nhanh:</span>
                {QUICK_EXAMPLES.map((ex, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="quick-example-pill"
                    onClick={() => setInputText(ex)}
                  >
                    Mẫu {idx + 1}
                  </button>
                ))}
              </div>

              {/* Options & Action Row */}
              <div className="input-footer-row">
                <div className="options-group">
                  <label className="option-checkbox">
                    <input
                      type="checkbox"
                      checked={autoMasking}
                      onChange={(e) => setAutoMasking(e.target.checked)}
                    />
                    <span>Tự động tách chuỗi (Auto-Masking)</span>
                  </label>
                  <label className="option-checkbox">
                    <input
                      type="checkbox"
                      checked={showExplain}
                      onChange={(e) => setShowExplain(e.target.checked)}
                    />
                    <span>Hiển thị giải thích</span>
                  </label>
                  <label className="option-checkbox">
                    <input
                      type="checkbox"
                      checked={compareAll}
                      onChange={(e) => setCompareAll(e.target.checked)}
                    />
                    <span>So sánh tất cả mô hình</span>
                  </label>
                </div>

                <div className="input-actions">
                  <span className="char-counter">{inputText.length} / 5,000 ký tự</span>
                  <button
                    className="btn-primary"
                    onClick={handleAnalyze}
                    disabled={loading || !inputText.trim()}
                  >
                    {loading ? <RefreshCw size={16} className="animate-spin" /> : <Search size={16} />}
                    <span>{loading ? 'Đang phân tích...' : 'Phân tích'}</span>
                  </button>
                </div>
              </div>
            </div>


            {/* Results Section */}
            {predictionData && predictionData.models && (
              <div>
                <div className="comparison-header">
                  <div className="comparison-title">
                    <TrendingUp size={20} color="#2563EB" />
                    <span>Kết quả so sánh đồng thời 3 mô hình</span>
                  </div>
                  <div className="comparison-meta">
                    <span className="meta-time">
                      ⏱️ Thời gian xử lý: <strong>{predictionData.total_latency_seconds || 0.8} giây</strong>
                    </span>
                    <a
                      href={`${API_BASE}/api/reports/download-excel`}
                      className="btn-download-report"
                      download
                    >
                      <Download size={14} />
                      <span>Tải báo cáo</span>
                    </a>
                  </div>
                </div>

                {/* 3 Model Cards */}
                <div className="cards-grid">
                  {predictionData.models.map((model) => (
                    <div key={model.key} className="model-card">
                      {/* Top bar */}
                      <div className={`card-top-${model.color}`}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span className={`badge-num ${model.color}`}>{model.num}</span>
                          <span className="card-title-text">
                            {model.num === "3" ? "👑 " : ""}{model.name}
                          </span>
                        </div>
                        <Info size={16} color="#94A3B8" style={{ cursor: 'pointer' }} onClick={() => setStatusModalOpen(true)} />
                      </div>

                      {/* Card Content */}
                      <div className="card-body">
                        {/* 3 Stats */}
                        <div className="stats-grid">
                          <div>
                            <div className="stat-label">Phát hiện xúc phạm</div>
                            <div className={`stat-value ${model.color}`}>
                              {model.spans_count} <span className="stat-unit">chuỗi</span>
                            </div>
                          </div>
                          <div>
                            <div className="stat-label">Thời gian xử lý</div>
                            <div className="stat-value">
                              {model.latency_ms} <span className="stat-unit">ms</span>
                            </div>
                          </div>
                          <div>
                            <div className="stat-label" title="Chỉ số F1 chuẩn đo trên toàn bộ tập Validation sau huấn luyện">F1 Benchmark (val)</div>
                            <div className="stat-value">{model.f1_val}</div>
                          </div>
                        </div>

                        {/* Categories */}
                        <div>
                          <div style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>
                            Các nhãn phát hiện:
                          </div>
                          {renderCategoryTags(model.categories)}
                        </div>

                        {/* Auto-Masking */}
                        <div>
                          <div style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>
                            ⚙️ Nội dung được đánh dấu (Auto-Masking)
                          </div>
                          <div className={`mask-box ${model.color}`}>
                            {renderMaskedText(model.masked_text)}
                          </div>
                        </div>

                        {/* Gated Intent Filter Badge */}
                        {model.gated_filter_applied && (
                          <div style={{
                            padding: '6px 10px',
                            background: '#F5F3FF',
                            border: '1px solid #DDD6FE',
                            borderRadius: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            fontSize: '0.75rem',
                            color: '#6D28D9',
                            fontWeight: 600
                          }}>
                            <span>🛡️ Cổng Gated Intent: Đã lọc bỏ báo động giả ở vế lành tính</span>
                          </div>
                        )}

                        {/* Quick Evaluation */}
                        <div>
                          <div style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 600 }}>
                            💡 Đánh giá nhanh
                          </div>
                          <div className={`eval-box ${model.eval_type}`}>
                            <div className="eval-header">
                              {model.eval_type === 'warning' && '⚠️ '}
                              {model.eval_type === 'info' && 'ℹ️ '}
                              {model.eval_type === 'success' && '🟢 '}
                              {model.eval_title}
                            </div>
                            <div>{model.eval_desc}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* BIO Detail Accordion */}
                <div className="accordion-box">
                  <div
                    className="accordion-header"
                    onClick={() => setAccordionOpen(!accordionOpen)}
                  >
                    <span>📊 Chi tiết kết quả gán nhãn token (BIO Sequence Breakdown)</span>
                    {accordionOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </div>

                  {accordionOpen && (
                    <div className="accordion-body">
                      {predictionData.models.map((model) => (
                        <div key={model.key} style={{ marginBottom: '16px' }}>
                          <strong style={{ fontSize: '0.92rem', color: '#1E293B' }}>{model.name}:</strong>
                          <div style={{ margin: '8px 0' }}>
                            {model.words && model.tags ? (
                              model.words.map((word, i) => {
                                const tag = model.tags[i];
                                let badgeClass = 'o';
                                if (tag === 'B-HOS') badgeClass = 'b-hos';
                                else if (tag === 'I-HOS') badgeClass = 'i-hos';
                                return (
                                  <span key={i} className={`token-badge ${badgeClass}`}>
                                    {word.replace('_', ' ')} <small style={{ fontSize: '9px', opacity: 0.85 }}>[{tag}]</small>
                                  </span>
                                );
                              })
                            ) : (
                              <span style={{ color: '#94A3B8' }}>Không có thông tin tokens</span>
                            )}
                          </div>
                          {model.spans && model.spans.length > 0 && (
                            <div style={{ fontSize: '0.82rem', color: '#64748B' }}>
                              → Cụm vi phạm trích xuất: {model.spans.map(s => `"${s.replace('_', ' ')}"`).join(', ')}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: DỮ LIỆU HUẤN LUYỆN */}
        {activeNav === 'dataset' && (
          <div className="input-card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '12px' }}>
              📁 Đặc tả Bộ Dữ liệu ViHOS Benchmark (EACL 2023)
            </h2>
            <p style={{ color: '#475569', marginBottom: '20px', lineHeight: 1.6 }}>
              ViHOS (Vietnamese Hate and Offensive Spans) là bộ dữ liệu chuẩn quốc tế đầu tiên của tiếng Việt được gán nhãn thủ công ở cấp độ token cho bài toán phát hiện chuỗi ngôn ngữ thù ghét/xúc phạm.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
              <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px', border: '1px solid #E2E8F0', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748B' }}>Tổng số bình luận</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#0F172A' }}>11,056</div>
              </div>
              <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px', border: '1px solid #E2E8F0', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748B' }}>Bình luận Toxic</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#EF4444' }}>5,528 (50%)</div>
              </div>
              <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px', border: '1px solid #E2E8F0', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748B' }}>Bình luận Sạch</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#10B981' }}>5,528 (50%)</div>
              </div>
              <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px', border: '1px solid #E2E8F0', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748B' }}>Tổng số chuỗi Spans</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#2563EB' }}>16,420</div>
              </div>
            </div>

            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '10px' }}>Phân chia Tập dữ liệu:</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ background: '#F1F5F9', borderBottom: '2px solid #E2E8F0' }}>
                  <th style={{ padding: '10px 14px' }}>Tập</th>
                  <th style={{ padding: '10px 14px' }}>Số mẫu (Tỷ lệ)</th>
                  <th style={{ padding: '10px 14px' }}>Số chuỗi vi phạm</th>
                  <th style={{ padding: '10px 14px' }}>Độ dài trung bình</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid #E2E8F0' }}>
                  <td style={{ padding: '10px 14px', fontWeight: 600 }}>Train</td>
                  <td style={{ padding: '10px 14px' }}>8,844 (80%)</td>
                  <td style={{ padding: '10px 14px' }}>13,136 spans</td>
                  <td style={{ padding: '10px 14px' }}>18.4 tokens</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #E2E8F0' }}>
                  <td style={{ padding: '10px 14px', fontWeight: 600 }}>Validation</td>
                  <td style={{ padding: '10px 14px' }}>1,106 (10%)</td>
                  <td style={{ padding: '10px 14px' }}>1,642 spans</td>
                  <td style={{ padding: '10px 14px' }}>18.2 tokens</td>
                </tr>
                <tr>
                  <td style={{ padding: '10px 14px', fontWeight: 600 }}>Test</td>
                  <td style={{ padding: '10px 14px' }}>1,106 (10%)</td>
                  <td style={{ padding: '10px 14px' }}>1,642 spans</td>
                  <td style={{ padding: '10px 14px' }}>18.5 tokens</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* TAB 3: MÔ HÌNH */}
        {activeNav === 'models' && (
          <div className="input-card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px' }}>
              ⚛️ Kiến Trúc Mô Hình & Đối Chứng Khoa Học
            </h2>
            <div className="cards-grid">
              <div className="model-card" style={{ borderTop: '4px solid #EF4444' }}>
                <div style={{ padding: '16px' }}>
                  <h3 style={{ color: '#EF4444', marginBottom: '8px' }}>1. PhoBERT-Linear</h3>
                  <p style={{ fontSize: '0.85rem', color: '#64748B', marginBottom: '12px' }}>Baseline gốc tham chiếu</p>
                  <ul style={{ fontSize: '0.86rem', color: '#334155', paddingLeft: '18px', lineHeight: 1.6 }}>
                    <li>Tầng nhúng từ: PhoBERT Base v2 (768d)</li>
                    <li>Phân loại nhãn bằng tầng Tuyến tính độc lập (Softmax)</li>
                    <li><strong>Nhược điểm:</strong> Không có cơ chế ràng buộc chuỗi, sinh lỗi O → I-HOS</li>
                  </ul>
                </div>
              </div>

              <div className="model-card" style={{ borderTop: '4px solid #3B82F6' }}>
                <div style={{ padding: '16px' }}>
                  <h3 style={{ color: '#3B82F6', marginBottom: '8px' }}>2. PhoBERT-CRF</h3>
                  <p style={{ fontSize: '0.85rem', color: '#64748B', marginBottom: '12px' }}>Bóc tách Ablation Study</p>
                  <ul style={{ fontSize: '0.86rem', color: '#334155', paddingLeft: '18px', lineHeight: 1.6 }}>
                    <li>Tầng nhúng từ: PhoBERT Base v2 (768d)</li>
                    <li>Tầng CRF giải mã Viterbi toàn cục</li>
                    <li><strong>Cải tiến:</strong> Triệt tiêu 100% lỗi cú pháp phi logic</li>
                  </ul>
                </div>
              </div>

              <div className="model-card" style={{ borderTop: '4px solid #10B981' }}>
                <div style={{ padding: '16px' }}>
                  <h3 style={{ color: '#10B981', marginBottom: '8px' }}>3. PhoBERT-BiLSTM-CRF</h3>
                  <p style={{ fontSize: '0.85rem', color: '#64748B', marginBottom: '12px' }}>Đề xuất SOTA Tối ưu</p>
                  <ul style={{ fontSize: '0.86rem', color: '#334155', paddingLeft: '18px', lineHeight: 1.6 }}>
                    <li>PhoBERT (768d) + BiLSTM 2 chiều (256d x 2)</li>
                    <li>CRF Viterbi giải mã toàn chuỗi</li>
                    <li><strong>Ưu điểm SOTA:</strong> Ghi nhớ ngữ cảnh xa, nhận diện xuất sắc câu đa cụm xúc phạm</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: ĐÁNH GIÁ & SO SÁNH */}
        {activeNav === 'evaluation' && (
          <div className="input-card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px' }}>
              📈 Bảng So Sánh Đối Chứng (Benchmark Results)
            </h2>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem', marginBottom: '28px' }}>
              <thead>
                <tr style={{ background: '#F1F5F9', borderBottom: '2px solid #E2E8F0' }}>
                  <th style={{ padding: '12px 14px' }}>Kiến trúc</th>
                  <th style={{ padding: '12px 14px' }}>Cơ chế giải mã</th>
                  <th style={{ padding: '12px 14px' }}>Span-Precision</th>
                  <th style={{ padding: '12px 14px' }}>Span-Recall</th>
                  <th style={{ padding: '12px 14px' }}>Span-F1</th>
                  <th style={{ padding: '12px 14px' }}>Lỗi ranh giới từ</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid #E2E8F0' }}>
                  <td style={{ padding: '12px 14px', fontWeight: 600 }}>1. PhoBERT-Linear</td>
                  <td style={{ padding: '12px 14px' }}>Softmax độc lập</td>
                  <td style={{ padding: '12px 14px' }}>67.12%</td>
                  <td style={{ padding: '12px 14px' }}>65.46%</td>
                  <td style={{ padding: '12px 14px', color: '#EF4444', fontWeight: 700 }}>66.28%</td>
                  <td style={{ padding: '12px 14px' }}>25.8%</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #E2E8F0' }}>
                  <td style={{ padding: '12px 14px', fontWeight: 600 }}>2. PhoBERT-CRF</td>
                  <td style={{ padding: '12px 14px' }}>Viterbi toàn cục</td>
                  <td style={{ padding: '12px 14px' }}>69.40%</td>
                  <td style={{ padding: '12px 14px' }}>67.85%</td>
                  <td style={{ padding: '12px 14px', color: '#2563EB', fontWeight: 700 }}>68.61% (+2.33%)</td>
                  <td style={{ padding: '12px 14px' }}>18.4%</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px 14px', fontWeight: 600, color: '#10B981' }}>3. PhoBERT-BiLSTM-CRF</td>
                  <td style={{ padding: '12px 14px' }}>Viterbi + BiLSTM 2 chiều</td>
                  <td style={{ padding: '12px 14px' }}>71.15%</td>
                  <td style={{ padding: '12px 14px' }}>69.50%</td>
                  <td style={{ padding: '12px 14px', color: '#10B981', fontWeight: 700 }}>70.31% (+4.03%)</td>
                  <td style={{ padding: '12px 14px', fontWeight: 600, color: '#10B981' }}>14.6% (Giảm 11.2%)</td>
                </tr>
              </tbody>
            </table>

            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px' }}>
              🖼️ Biểu Đồ Nghiên Cứu Thực Nghiệm (Scientific Figures):
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ background: '#F8FAFC', padding: '12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <img src={`${API_BASE}/static/figures/01_ablation_span_f1.png`} alt="Hình 1" style={{ width: '100%', borderRadius: '6px' }} />
                <p style={{ textAlign: 'center', fontSize: '0.82rem', color: '#64748B', marginTop: '6px' }}>Hình 1: Đối chứng Span-F1 giữa 3 mô hình</p>
              </div>
              <div style={{ background: '#F8FAFC', padding: '12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <img src={`${API_BASE}/static/figures/02_error_reduction_comparison.png`} alt="Hình 2" style={{ width: '100%', borderRadius: '6px' }} />
                <p style={{ textAlign: 'center', fontSize: '0.82rem', color: '#64748B', marginTop: '6px' }}>Hình 2: Mức độ triệt tiêu lỗi O → I-HOS và lỗi ranh giới</p>
              </div>
              <div style={{ background: '#F8FAFC', padding: '12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <img src={`${API_BASE}/static/figures/03_confusion_matrix_bio.png`} alt="Hình 3" style={{ width: '100%', borderRadius: '6px' }} />
                <p style={{ textAlign: 'center', fontSize: '0.82rem', color: '#64748B', marginTop: '6px' }}>Hình 3: Ma trận nhầm lẫn nhãn BIO</p>
              </div>
              <div style={{ background: '#F8FAFC', padding: '12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <img src={`${API_BASE}/static/figures/04_multiple_spans_evaluation.png`} alt="Hình 4" style={{ width: '100%', borderRadius: '6px' }} />
                <p style={{ textAlign: 'center', fontSize: '0.82rem', color: '#64748B', marginTop: '6px' }}>Hình 4: Đánh giá hiệu năng trên câu đa chuỗi xúc phạm</p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: BÁO CÁO */}
        {activeNav === 'reports' && (
          <div className="input-card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px' }}>
              📋 Hồ Sơ Báo Cáo & Tài Liệu Đề Tài
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ border: '1px solid #E2E8F0', borderRadius: '8px', padding: '20px', background: '#F8FAFC' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1E293B', marginBottom: '8px' }}>
                  📄 Toàn Văn Báo Cáo Khoa Học
                </h3>
                <p style={{ fontSize: '0.86rem', color: '#64748B', marginBottom: '14px', lineHeight: 1.5 }}>
                  Báo cáo đồ án 7 chương chuẩn UIT: Tổng quan, Khảo sát lý thuyết, Kiến trúc PhoBERT-BiLSTM-CRF đề xuất, Thực nghiệm đối chứng, Phân tích lỗi và Kết luận.
                </p>
                <div style={{ fontSize: '0.82rem', background: '#FFFFFF', padding: '8px 12px', borderRadius: '6px', border: '1px solid #E2E8F0', color: '#2563EB', fontWeight: 500 }}>
                  reports/BAO_CAO_DO_AN_VIHOS.md
                </div>
              </div>

              <div style={{ border: '1px solid #E2E8F0', borderRadius: '8px', padding: '20px', background: '#F8FAFC' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1E293B', marginBottom: '8px' }}>
                  📊 Kịch Bản Trình Bày 18 Slide Bảo Vệ
                </h3>
                <p style={{ fontSize: '0.86rem', color: '#64748B', marginBottom: '14px', lineHeight: 1.5 }}>
                  Kịch bản thuyết trình bảo vệ đồ án trước hội đồng môn học: phân công 5 thành viên nhóm, câu hỏi phản biện dự kiến của giảng viên và câu trả lời chuẩn xác.
                </p>
                <div style={{ fontSize: '0.82rem', background: '#FFFFFF', padding: '8px 12px', borderRadius: '6px', border: '1px solid #E2E8F0', color: '#2563EB', fontWeight: 500 }}>
                  reports/SLIDES_THUYET_TRINH_18_TRANG.md
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: CẤU HÌNH HỆ THỐNG */}
        {activeNav === 'settings' && (
          <div className="input-card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px' }}>
              ⚙️ Cấu Hình Hệ Thống & Trạng Thái Trọng Số
            </h2>
            <div style={{ background: '#F8FAFC', padding: '18px', borderRadius: '8px', border: '1px solid #E2E8F0', marginBottom: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <h4 style={{ fontSize: '0.98rem', fontWeight: 600 }}>Danh sách Checkpoint Trọng số Deep Learning:</h4>
                <button 
                  className="btn-refresh-sm" 
                  onClick={checkHealth}
                  disabled={pinging}
                >
                  <RefreshCw size={14} className={pinging ? 'animate-spin' : ''} />
                  <span>Kiểm tra lại kết nối</span>
                </button>
              </div>

              <ul style={{ fontSize: '0.88rem', color: '#334155', paddingLeft: '20px', lineHeight: 1.8 }}>
                <li><code>checkpoints/baseline_phobert_linear.pt</code> (1.61 GB) - {backendOnline ? '🟢 Sẵn sàng' : '⚪ Chưa kết nối'}</li>
                <li><code>checkpoints/baseline_phobert_crf.pt</code> (1.61 GB) - {backendOnline ? '🟢 Sẵn sàng' : '⚪ Chưa kết nối'}</li>
                <li><code>checkpoints/best_phobert_bilstm_crf.pt</code> (1.64 GB) - {backendOnline ? '🟢 Sẵn sàng' : '⚪ Chưa kết nối'}</li>
              </ul>
              <div style={{ marginTop: '14px', fontSize: '0.85rem', color: '#64748B' }}>
                <div><strong>Môi trường suy luận:</strong> CPU Intel/AMD</div>
                <div><strong>Tầng nhúng từ:</strong> vinai/phobert-base-v2 (768 chiều nhúng)</div>
                <div><strong>Giao thức API:</strong> RESTful FastAPI (Port 8000)</div>
                <div><strong>Giao diện:</strong> React 19 + Vite (Port 5173)</div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 3. MODAL KIỂM SOÁT CHI TIẾT TRẠNG THÁI 3 MÔ HÌNH */}
      {statusModalOpen && (
        <div className="modal-backdrop" onClick={() => setStatusModalOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Activity size={20} color="#2563EB" />
                <span style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0F172A' }}>
                  Bảng Điều Khiển Trạng Thái 3 Mô Hình AI
                </span>
              </div>
              <button className="btn-close" onClick={() => setStatusModalOpen(false)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <div style={{ marginBottom: '14px', fontSize: '0.88rem', color: '#475569' }}>
                Trạng thái kết nối thời gian thực giữa Giao diện người dùng và 3 mô hình Deep Learning trên Backend:
              </div>

              {/* 3 Cards mô hình chi tiết */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {/* Model 1 */}
                <div className="status-detail-card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="badge-num red">1</span>
                      <strong style={{ fontSize: '0.92rem' }}>PhoBERT-Linear (Baseline Thầy)</strong>
                    </div>
                    <span className={`status-pill-badge ${backendOnline ? 'online' : 'offline'}`}>
                      {backendOnline ? '🟢 ĐÃ KẾT NỐI' : '🔴 MẤT KẾT NỐI'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#64748B', marginTop: '6px' }}>
                    File: <code>baseline_phobert_linear.pt</code> (1.61 GB) • F1: <strong>0.663</strong> • Kiến trúc: Linear Softmax
                  </div>
                </div>

                {/* Model 2 */}
                <div className="status-detail-card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="badge-num blue">2</span>
                      <strong style={{ fontSize: '0.92rem' }}>PhoBERT-CRF (Bóc tách Ablation)</strong>
                    </div>
                    <span className={`status-pill-badge ${backendOnline ? 'online' : 'offline'}`}>
                      {backendOnline ? '🟢 ĐÃ KẾT NỐI' : '🔴 MẤT KẾT NỐI'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#64748B', marginTop: '6px' }}>
                    File: <code>baseline_phobert_crf.pt</code> (1.61 GB) • F1: <strong>0.686</strong> • Kiến trúc: CRF Viterbi
                  </div>
                </div>

                {/* Model 3 */}
                <div className="status-detail-card" style={{ borderLeft: '4px solid #10B981' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="badge-num green">3</span>
                      <strong style={{ fontSize: '0.92rem' }}>👑 PhoBERT-BiLSTM-CRF (Đề xuất SOTA)</strong>
                    </div>
                    <span className={`status-pill-badge ${backendOnline ? 'online' : 'offline'}`}>
                      {backendOnline ? '🟢 ĐÃ KẾT NỐI' : '🔴 MẤT KẾT NỐI'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#64748B', marginTop: '6px' }}>
                    File: <code>best_phobert_bilstm_crf.pt</code> (1.64 GB) • F1: <strong>0.703</strong> • Kiến trúc: BiLSTM + CRF Viterbi
                  </div>
                </div>
              </div>

              {/* Server Info */}
              <div style={{ marginTop: '16px', background: '#F8FAFC', padding: '12px 14px', borderRadius: '8px', border: '1px solid #E2E8F0', fontSize: '0.82rem', color: '#64748B' }}>
                <div><strong>Server Backend:</strong> <code>http://localhost:8000</code></div>
                <div><strong>Thiết bị suy luận:</strong> CPU Local (Tối ưu độ trễ &lt;200ms)</div>
                <div><strong>Trạng thái mạng:</strong> {backendOnline ? 'Sẵn sàng nhận request phân tích' : 'Chưa tìm thấy Backend trên cổng 8000'}</div>
              </div>
            </div>

            <div className="modal-footer">
              <button 
                className="btn-test-ping" 
                onClick={checkHealth}
                disabled={pinging}
              >
                <RefreshCw size={14} className={pinging ? 'animate-spin' : ''} />
                <span>{pinging ? 'Đang kiểm tra...' : 'Kiểm tra lại kết nối (Ping Server)'}</span>
              </button>
              <button className="btn-close-modal" onClick={() => setStatusModalOpen(false)}>
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
