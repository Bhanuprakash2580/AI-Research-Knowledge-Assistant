from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import documents, search, analysis, memory, analytics
from .db import initialize_database

app = FastAPI(
    title="AI Research & Knowledge Assistant",
    description="Grounded QA, Document Analysis, Semantic Search & Knowledge Base Analytics",
    version="1.0.0"
)

# Enable CORS for cross-origin deployments (e.g. Vercel frontend <-> Render backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
try:
    initialize_database()
except Exception as e:
    print(f"Database initialization warning: {e}")

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(memory.router)
app.include_router(analytics.router)

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Research & Knowledge Assistant</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 29, 47, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --accent: #8b5cf6;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --radius: 12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; background-image: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15), transparent 50%); }
    header { border-bottom: 1px solid var(--card-border); backdrop-filter: blur(12px); background: rgba(9, 13, 22, 0.8); position: sticky; top: 0; z-index: 100; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
    .logo { display: flex; align-items: center; gap: 0.75rem; font-weight: 700; font-size: 1.25rem; background: linear-gradient(135deg, #a5b4fc, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .badge-status { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); font-size: 0.75rem; padding: 0.25rem 0.65rem; border-radius: 999px; display: inline-flex; align-items: center; gap: 0.35rem; font-weight: 500; }
    .badge-status::before { content: ''; width: 6px; height: 6px; background: var(--success); border-radius: 50%; display: inline-block; box-shadow: 0 0 8px var(--success); }
    nav-links { display: flex; gap: 1rem; }
    .btn-link { color: var(--text-muted); text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: all 0.2s; padding: 0.4rem 0.8rem; border-radius: 6px; }
    .btn-link:hover { color: var(--text); background: rgba(255,255,255,0.05); }

    main { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; flex: 1; width: 100%; }
    
    .nav-tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem; flex-wrap: wrap; }
    .tab-btn { background: transparent; border: none; color: var(--text-muted); padding: 0.65rem 1.25rem; font-weight: 500; font-size: 0.925rem; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
    .tab-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.04); }
    .tab-btn.active { color: #fff; background: var(--primary); box-shadow: 0 4px 14px var(--primary-glow); }

    .tab-content { display: none; animation: fadeIn 0.3s ease; }
    .tab-content.active { display: block; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; backdrop-filter: blur(8px); }
    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #fff; display: flex; justify-content: space-between; align-items: center; }

    .upload-area { border: 2px dashed rgba(99, 102, 241, 0.4); border-radius: var(--radius); padding: 2.5rem; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(99, 102, 241, 0.02); }
    .upload-area:hover { border-color: var(--primary); background: rgba(99, 102, 241, 0.06); }

    .btn { background: var(--primary); color: white; border: none; padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 500; cursor: pointer; transition: all 0.2s; font-size: 0.9rem; display: inline-flex; align-items: center; gap: 0.5rem; }
    .btn:hover { background: var(--primary-hover); box-shadow: 0 4px 12px var(--primary-glow); }
    .btn-secondary { background: rgba(255,255,255,0.08); color: var(--text); }
    .btn-secondary:hover { background: rgba(255,255,255,0.15); }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
    .btn-danger:hover { background: var(--danger); color: white; }

    input[type="text"], select, textarea { width: 100%; background: rgba(0, 0, 0, 0.3); border: 1px solid var(--card-border); border-radius: 8px; padding: 0.75rem 1rem; color: white; font-size: 0.9rem; outline: none; transition: border 0.2s; }
    input[type="text"]:focus, select:focus, textarea:focus { border-color: var(--primary); }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }

    .stat-box { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--card-border); border-radius: 10px; padding: 1.2rem; text-align: center; }
    .stat-val { font-size: 1.8rem; font-weight: 700; color: var(--primary); margin-top: 0.25rem; }
    .stat-lbl { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

    table { width: 100%; border-collapse: collapse; margin-top: 1rem; text-align: left; }
    th { padding: 0.75rem 1rem; color: var(--text-muted); font-size: 0.8rem; border-bottom: 1px solid var(--card-border); text-transform: uppercase; font-weight: 600; }
    td { padding: 0.9rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.9rem; }
    
    .chip { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }

    .citation-card { background: rgba(0, 0, 0, 0.25); border-left: 3px solid var(--primary); padding: 0.8rem 1rem; border-radius: 4px; margin-top: 0.75rem; font-size: 0.85rem; }
    
    footer { text-align: center; padding: 2rem; border-top: 1px solid var(--card-border); color: var(--text-muted); font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      AI Research & Knowledge Assistant
    </div>
    <div style="display: flex; align-items: center; gap: 1.5rem;">
      <span class="badge-status">API Operational</span>
      <a href="/docs" class="btn-link" target="_blank">Interactive Swagger Docs ↗</a>
    </div>
  </header>

  <main>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('documents')">📄 Document Management</button>
      <button class="tab-btn" onclick="switchTab('search')">🔍 Grounded Search & QA</button>
      <button class="tab-btn" onclick="switchTab('analysis')">📊 Summarize & Compare</button>
      <button class="tab-btn" onclick="switchTab('analytics')">📈 System Analytics</button>
    </div>

    <!-- TAB 1: DOCUMENTS -->
    <div id="tab-documents" class="tab-content active">
      <div class="card">
        <div class="card-title">Upload Research Documents (PDF)</div>
        <div class="upload-area" onclick="document.getElementById('pdf-file-input').click()">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:0.5rem; color:var(--primary)"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
          <p style="font-weight: 500;">Click to select or drag and drop PDF files</p>
          <p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.3rem;">Supports text extraction, page-aware chunking, & automated TF/heuristic classification</p>
          <input type="file" id="pdf-file-input" accept=".pdf" style="display:none" onchange="uploadDocument(this.files[0])">
        </div>
        <div id="upload-status" style="margin-top: 0.75rem; font-size: 0.85rem;"></div>
      </div>

      <div class="card">
        <div class="card-title">
          Uploaded Knowledge Base
          <button class="btn btn-secondary" onclick="loadDocuments()">↻ Refresh</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>Document Name</th>
              <th>Category</th>
              <th>Pages</th>
              <th>Chunks</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="docs-list">
            <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Loading documents...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 2: SEARCH & QA -->
    <div id="tab-search" class="tab-content">
      <div class="card">
        <div class="card-title">Ask Grounded Questions or Search</div>
        <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem;">
          <input type="text" id="qa-query" placeholder="e.g. What are the key findings or methodologies presented in the papers?">
          <button class="btn" onclick="executeQA()">Ask Question</button>
        </div>
        <div style="display: flex; gap: 1.5rem; font-size: 0.85rem; color: var(--text-muted); align-items: center; flex-wrap: wrap;">
          <label>Mode: 
            <select id="qa-mode" style="width: auto; padding: 0.3rem 0.6rem; margin-left: 0.3rem;">
              <option value="hybrid">Hybrid (Keyword + Semantic)</option>
              <option value="semantic">Semantic</option>
              <option value="keyword">Keyword</option>
            </select>
          </label>
          <label>Top Context Chunks (k): 
            <input type="number" id="qa-k" value="5" min="1" max="20" style="width: 70px; padding: 0.3rem 0.5rem; margin-left: 0.3rem;">
          </label>
          <label>Session ID: 
            <input type="text" id="qa-session" value="web-session" style="width: 140px; padding: 0.3rem 0.5rem; margin-left: 0.3rem;">
          </label>
        </div>
      </div>

      <div class="card" id="qa-result-card" style="display: none;">
        <div class="card-title">Grounded Answer & Source Citations</div>
        <div id="qa-answer" style="background: rgba(99,102,241,0.08); padding: 1.2rem; border-radius: 8px; line-height: 1.6; margin-bottom: 1rem; border: 1px solid rgba(99,102,241,0.2);"></div>
        <h4 style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem;">Retrieved Context Citations:</h4>
        <div id="qa-citations"></div>
      </div>
    </div>

    <!-- TAB 3: ANALYSIS -->
    <div id="tab-analysis" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Document Summarization</div>
          <div style="margin-bottom: 1rem;">
            <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Select Document:</label>
            <select id="sum-doc-select"></select>
          </div>
          <div style="margin-bottom: 1rem;">
            <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Summary Type:</label>
            <select id="sum-type">
              <option value="executive">Executive Summary</option>
              <option value="technical">Technical Summary</option>
              <option value="bullets">Bullet Points</option>
              <option value="takeaways">Key Takeaways</option>
            </select>
          </div>
          <button class="btn" onclick="generateSummary()">Generate Summary</button>
          <div id="summary-output" style="margin-top: 1rem;"></div>
        </div>

        <div class="card">
          <div class="card-title">Compare Documents</div>
          <div style="margin-bottom: 1rem;">
            <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Focus Area:</label>
            <input type="text" id="comp-focus" value="methodologies and key findings">
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.8rem;">Select 2 or more documents from your library to synthesize similarities and differences.</p>
          <div id="comp-docs-checklist" style="max-height: 160px; overflow-y: auto; margin-bottom: 1rem;"></div>
          <button class="btn btn-secondary" onclick="compareDocuments()">Run Comparative Analysis</button>
          <div id="comparison-output" style="margin-top: 1rem;"></div>
        </div>
      </div>
    </div>

    <!-- TAB 4: ANALYTICS -->
    <div id="tab-analytics" class="tab-content">
      <div class="grid-4" style="margin-bottom: 1.5rem;">
        <div class="stat-box"><div class="stat-lbl">Total Documents</div><div class="stat-val" id="stat-docs">0</div></div>
        <div class="stat-box"><div class="stat-lbl">Total Text Chunks</div><div class="stat-val" id="stat-chunks">0</div></div>
        <div class="stat-box"><div class="stat-lbl">Total Queries Run</div><div class="stat-val" id="stat-queries">0</div></div>
        <div class="stat-box"><div class="stat-lbl">Grounded Answers</div><div class="stat-val" id="stat-answers">0</div></div>
      </div>

      <div class="card">
        <div class="card-title">Knowledge Base Distribution & Metrics</div>
        <div id="analytics-details">Loading stats...</div>
      </div>
    </div>
  </main>

  <footer>
    AI Research & Knowledge Assistant &bull; Production Backend & Interactive Dashboard
  </footer>

  <script>
    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + name).classList.add('active');
      if (name === 'documents') loadDocuments();
      if (name === 'analysis') populateAnalysisSelectors();
      if (name === 'analytics') loadAnalytics();
    }

    async function loadDocuments() {
      try {
        const res = await fetch('/documents/');
        const docs = await res.json();
        const tbody = document.getElementById('docs-list');
        if (!docs.length) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted);">No documents uploaded yet. Upload a PDF above!</td></tr>';
          return;
        }
        tbody.innerHTML = docs.map(d => `
          <tr>
            <td><strong>${d.name}</strong></td>
            <td><span class="chip">${d.category || 'Unclassified'}</span></td>
            <td>${d.total_pages || 0}</td>
            <td>${d.total_chunks || 0}</td>
            <td><span style="color: ${d.status === 'processed' ? 'var(--success)' : 'var(--warning)'}">${d.status}</span></td>
            <td>
              <button class="btn btn-secondary" style="padding: 0.25rem 0.5rem; font-size:0.75rem;" onclick="reprocessDoc('${d.id}')">Reprocess</button>
              <button class="btn btn-danger" style="padding: 0.25rem 0.5rem; font-size:0.75rem;" onclick="deleteDoc('${d.id}')">Delete</button>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    async function uploadDocument(file) {
      if (!file) return;
      const statusDiv = document.getElementById('upload-status');
      statusDiv.innerHTML = `<span style="color:var(--warning)">Uploading ${file.name}...</span>`;
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/documents/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          statusDiv.innerHTML = `<span style="color:var(--success)">✓ Successfully uploaded & queued processing!</span>`;
          loadDocuments();
        } else {
          statusDiv.innerHTML = `<span style="color:var(--danger)">Upload failed: ${data.detail || 'Error'}</span>`;
        }
      } catch (err) {
        statusDiv.innerHTML = `<span style="color:var(--danger)">Error: ${err.message}</span>`;
      }
    }

    async function deleteDoc(id) {
      if (!confirm('Are you sure you want to delete this document?')) return;
      await fetch('/documents/' + id, { method: 'DELETE' });
      loadDocuments();
    }

    async function reprocessDoc(id) {
      await fetch('/documents/' + id + '/reprocess', { method: 'POST' });
      loadDocuments();
    }

    async function executeQA() {
      const query = document.getElementById('qa-query').value;
      if (!query) return;
      const mode = document.getElementById('qa-mode').value;
      const k = parseInt(document.getElementById('qa-k').value);
      const session_id = document.getElementById('qa-session').value;

      const resCard = document.getElementById('qa-result-card');
      const ansDiv = document.getElementById('qa-answer');
      const citDiv = document.getElementById('qa-citations');
      
      resCard.style.display = 'block';
      ansDiv.innerHTML = 'Searching knowledge base and generating answer...';
      citDiv.innerHTML = '';

      try {
        const res = await fetch('/search/qa', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, mode, k, session_id })
        });
        const data = await res.json();
        ansDiv.innerHTML = `<strong>Answer:</strong><br>${data.answer || 'No answer generated.'}`;
        if (data.context && data.context.length) {
          citDiv.innerHTML = data.context.map(c => `
            <div class="citation-card">
              <strong>${c.document}</strong> (Page ${c.page_number || 'N/A'}) &bull; Score: ${(c.score || 0).toFixed(2)}
              <p style="color: var(--text-muted); margin-top:0.3rem;">"${c.text}"</p>
            </div>
          `).join('');
        } else {
          citDiv.innerHTML = '<p style="color:var(--text-muted)">No matching citations found in index.</p>';
        }
      } catch (err) {
        ansDiv.innerHTML = '<span style="color:var(--danger)">Error running QA query.</span>';
      }
    }

    async function populateAnalysisSelectors() {
      const res = await fetch('/documents/');
      const docs = await res.json();
      const sumSelect = document.getElementById('sum-doc-select');
      const checklist = document.getElementById('comp-docs-checklist');

      sumSelect.innerHTML = docs.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
      checklist.innerHTML = docs.map(d => `
        <label style="display:block; margin-bottom: 0.3rem;">
          <input type="checkbox" class="comp-check" value="${d.id}"> ${d.name}
        </label>
      `).join('');
    }

    async function generateSummary() {
      const docId = document.getElementById('sum-doc-select').value;
      const type = document.getElementById('sum-type').value;
      const out = document.getElementById('summary-output');
      if (!docId) return out.innerHTML = 'Select a document first.';
      out.innerHTML = 'Generating summary...';
      try {
        const res = await fetch(`/analysis/summarize/${docId}?type=${type}`);
        const data = await res.json();
        out.innerHTML = `<div style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px; white-space:pre-wrap;">${JSON.stringify(data.summary, null, 2)}</div>`;
      } catch (err) {
        out.innerHTML = 'Failed to generate summary.';
      }
    }

    async function compareDocuments() {
      const focus = document.getElementById('comp-focus').value;
      const selected = Array.from(document.querySelectorAll('.comp-check:checked')).map(c => c.value);
      const out = document.getElementById('comparison-output');
      if (selected.length < 2) return out.innerHTML = '<span style="color:var(--warning)">Select at least 2 documents to compare.</span>';
      out.innerHTML = 'Analyzing and synthesizing comparison...';
      try {
        const res = await fetch('/analysis/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doc_ids: selected, focus })
        });
        const data = await res.json();
        out.innerHTML = `<div style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px; white-space:pre-wrap;">${JSON.stringify(data.comparison, null, 2)}</div>`;
      } catch (err) {
        out.innerHTML = 'Comparison failed.';
      }
    }

    async function loadAnalytics() {
      try {
        const res = await fetch('/analytics/stats');
        const data = await res.json();
        document.getElementById('stat-docs').innerText = data.total_documents || 0;
        document.getElementById('stat-chunks').innerText = data.total_chunks || 0;
        document.getElementById('stat-queries').innerText = data.query_stats?.total_queries || 0;
        document.getElementById('stat-answers').innerText = data.query_stats?.total_questions_answered || 0;

        document.getElementById('analytics-details').innerHTML = `
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1.5rem;">
            <div>
              <h4 style="margin-bottom:0.5rem; color:var(--text-muted);">Category Distribution</h4>
              <pre style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px;">${JSON.stringify(data.category_distribution || {}, null, 2)}</pre>
            </div>
            <div>
              <h4 style="margin-bottom:0.5rem; color:var(--text-muted);">Most Queried Documents</h4>
              <pre style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px;">${JSON.stringify(data.query_stats?.most_queried_documents || [], null, 2)}</pre>
            </div>
          </div>
        `;
      } catch (err) {
        console.error(err);
      }
    }

    // Initial load
    loadDocuments();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept or request.query_params.get("ui"):
        return HTMLResponse(content=HTML_UI)
    return JSONResponse({"status": "ok", "message": "AI Research & Knowledge Assistant backend"})

