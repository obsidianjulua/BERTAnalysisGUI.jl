module EngineGUI
include("EngineT.jl")

using Blink
using JSON3
using .EngineT

export launch_gui

# Global state
mutable struct AppState
    engine::Union{Nothing,Engine}
    current_theme::String
    AppState() = new(nothing, "dark")
end

const state = AppState()

"""
Launch the EngineT GUI application
"""
function launch_gui(model_name::String="bert-base-uncased")
    println("📦 Loading model: $model_name")

    # Initialize engine
    try
        state.engine = Engine(model_name)
        println("✓ Engine loaded successfully")
    catch e
        println("✗ Error loading engine: $e")
        println("GUI will launch but functionality will be limited")
    end

    # Create window
    w = Window(Dict("title" => "EngineT - BERT Text Analysis",
        "width" => 1200,
        "height" => 800))

    # Load HTML interface
    body!(w, html_content())

    # Setup handlers
    setup_handlers(w)

    println("✓ GUI launched successfully!")
    println("📱 Window should now be open")

    return w
end

"""
Generate the HTML content for the GUI
"""
function html_content()
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>EngineT GUI</title>
    <style>
    $(css_styles())
    </style>
    </head>
    <body>
    <div class="container">
    <header>
    <h1>🧠 EngineT - BERT Text Analysis</h1>
    <button id="theme-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
    </header>

    <div class="tabs">
    <button class="tab-button active" onclick="openTab('search')">🔍 Semantic Search</button>
    <button class="tab-button" onclick="openTab('similarity')">📊 Text Similarity</button>
    <button class="tab-button" onclick="openTab('classification')">🏷️ Classification</button>
    <button class="tab-button" onclick="openTab('clustering')">🗂️ Clustering</button>
    <button class="tab-button" onclick="openTab('analysis')">🔬 Text Analysis</button>
    </div>

    <!-- Semantic Search Tab -->
    <div id="search" class="tab-content active">
    <h2>🔍 Semantic Search</h2>
    <div class="input-section">
    <label>Query:</label>
    <textarea id="search-query" rows="2" placeholder="Enter your search query..."></textarea>

    <label>Documents (one per line):</label>
    <textarea id="search-documents" rows="8" placeholder="Enter documents, one per line..."></textarea>

    <label>Top K Results:</label>
    <input type="number" id="search-top-k" value="5" min="1" max="50">

    <button onclick="performSemanticSearch()">🔍 Search</button>
    </div>
    <div id="search-results" class="results-section"></div>
    </div>

    <!-- Text Similarity Tab -->
    <div id="similarity" class="tab-content">
    <h2>📊 Text Similarity</h2>
    <div class="split-panel">
    <div class="panel">
    <h3>Text 1</h3>
    <textarea id="sim-text1" rows="6" placeholder="Enter first text..."></textarea>
    </div>
    <div class="panel">
    <h3>Text 2</h3>
    <textarea id="sim-text2" rows="6" placeholder="Enter second text..."></textarea>
    </div>
    </div>
    <button onclick="compareSimilarity()">📊 Compare</button>
    <div id="similarity-results" class="results-section"></div>
    </div>

    <!-- Classification Tab -->
    <div id="classification" class="tab-content">
    <h2>🏷️ Text Classification</h2>
    <div class="input-section">
    <label>Text to Classify:</label>
    <textarea id="classify-text" rows="4" placeholder="Enter text to classify..."></textarea>

    <label>Classification Mode:</label>
    <select id="classify-mode">
    <option value="sentiment">Sentiment Analysis</option>
    <option value="topic">Topic Detection</option>
    </select>

    <button onclick="classifyText()">🏷️ Classify</button>
    </div>
    <div id="classification-results" class="results-section"></div>
    </div>

    <!-- Clustering Tab -->
    <div id="clustering" class="tab-content">
    <h2>🗂️ Text Clustering</h2>
    <div class="input-section">
    <label>Texts to Cluster (one per line, minimum 2):</label>
    <textarea id="cluster-texts" rows="10" placeholder="Enter texts, one per line..."></textarea>

    <label>Similarity Threshold:</label>
    <input type="range" id="cluster-threshold" min="0" max="1" step="0.05" value="0.7">
    <span id="cluster-threshold-value">0.7</span>

    <div style="margin-top: 20px;">
    <button onclick="performClustering()">🗂️ Cluster</button>
    <button onclick="findOutliers()">🎯 Find Outliers</button>
    </div>
    </div>
    <div id="clustering-results" class="results-section"></div>
    </div>

    <!-- Text Analysis Tab -->
    <div id="analysis" class="tab-content">
    <h2>🔬 Text Analysis</h2>
    <div class="input-section">
    <label>Text to Analyze:</label>
    <textarea id="analysis-text" rows="6" placeholder="Enter text to analyze..."></textarea>

    <button onclick="analyzeComplexity()">📈 Analyze Complexity</button>
    <button onclick="analyzeCoherence()">🔗 Analyze Coherence</button>
    <button onclick="analyzeReadability()">📖 Analyze Readability</button>
    <button onclick="analyzeTokens()">🔤 Token Analysis</button>
    </div>
    <div id="analysis-results" class="results-section"></div>
    </div>

    <footer>
    <p>EngineT v1.0 | Powered by BERT & Julia</p>
    </footer>
    </div>

    <script>
    $(javascript_code())
    </script>
    </body>
    </html>
    """
end

"""
CSS Styles for the GUI
"""
function css_styles()
    return """
    :root {
        --bg-primary: #1e1e1e;
        --bg-secondary: #2d2d2d;
        --bg-tertiary: #3d3d3d;
        --text-primary: #e0e0e0;
        --text-secondary: #b0b0b0;
        --accent: #4a9eff;
        --accent-hover: #66b3ff;
        --border: #444;
        --success: #4caf50;
        --error: #f44336;
    }

    body.light-theme {
        --bg-primary: #ffffff;
        --bg-secondary: #f5f5f5;
        --bg-tertiary: #e0e0e0;
        --text-primary: #212121;
        --text-secondary: #616161;
        --border: #ddd;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: all 0.3s ease;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }

    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 2px solid var(--border);
        margin-bottom: 20px;
    }

    h1 {
        font-size: 2rem;
        color: var(--accent);
    }

    h2 {
        margin-bottom: 20px;
        color: var(--accent);
    }

    h3 {
        margin: 15px 0 10px 0;
        color: var(--text-primary);
    }

    .tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--border);
        flex-wrap: wrap;
    }

    .tab-button {
        padding: 12px 24px;
        background: var(--bg-secondary);
        border: none;
        border-bottom: 3px solid transparent;
        color: var(--text-secondary);
        cursor: pointer;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .tab-button:hover {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }

    .tab-button.active {
        border-bottom-color: var(--accent);
        color: var(--accent);
        background: var(--bg-tertiary);
    }

    .tab-content {
        display: none;
        animation: fadeIn 0.3s ease;
    }

    .tab-content.active {
        display: block;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .input-section {
        background: var(--bg-secondary);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    label {
        display: block;
        margin: 15px 0 5px 0;
        font-weight: 600;
        color: var(--text-primary);
    }

    textarea, input[type="text"], input[type="number"], select {
        width: 100%;
        padding: 12px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-size: 0.95rem;
        font-family: 'Consolas', 'Monaco', monospace;
        resize: vertical;
    }

    textarea:focus, input:focus, select:focus {
        outline: none;
        border-color: var(--accent);
    }

    button {
        padding: 12px 24px;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 600;
        margin: 10px 5px 0 0;
        transition: all 0.2s ease;
    }

    button:hover {
        background: var(--accent-hover);
        transform: translateY(-2px);
    }

    button:active {
        transform: translateY(0);
    }

    #theme-toggle {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 8px 16px;
        font-size: 0.9rem;
    }

    .results-section {
        background: var(--bg-secondary);
        padding: 20px;
        border-radius: 8px;
        min-height: 200px;
    }

    .result-card {
        background: var(--bg-primary);
        padding: 15px;
        margin: 10px 0;
        border-radius: 6px;
        border-left: 4px solid var(--accent);
    }

    .result-card h4 {
        color: var(--accent);
        margin-bottom: 8px;
    }

    .result-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border);
    }

    .result-item:last-child {
        border-bottom: none;
    }

    .score {
        font-weight: bold;
        color: var(--accent);
    }

    .split-panel {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }

    .panel {
        background: var(--bg-secondary);
        padding: 20px;
        border-radius: 8px;
    }

    input[type="range"] {
        width: calc(100% - 60px);
        margin-right: 10px;
    }

    .progress-bar {
        width: 100%;
        height: 24px;
        background: var(--bg-tertiary);
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent), var(--accent-hover));
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }

    footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 2px solid var(--border);
        color: var(--text-secondary);
    }

    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border);
        border-top-color: var(--accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    @media (max-width: 768px) {
        .split-panel {
            grid-template-columns: 1fr;
        }
        .tabs {
            flex-direction: column;
        }
    }
    """
end

"""
JavaScript code for GUI interactions
"""
function javascript_code()
    return """
    // Tab switching
    function openTab(tabName) {
        const tabs = document.querySelectorAll('.tab-content');
        const buttons = document.querySelectorAll('.tab-button');

        tabs.forEach(tab => tab.classList.remove('active'));
        buttons.forEach(btn => btn.classList.remove('active'));

        document.getElementById(tabName).classList.add('active');
        event.target.classList.add('active');
    }

    // Theme toggle
    function toggleTheme() {
        document.body.classList.toggle('light-theme');
        Blink.msg('toggle_theme');
    }

    // Update threshold displays
    document.getElementById('cluster-threshold')?.addEventListener('input', (e) => {
        document.getElementById('cluster-threshold-value').textContent = e.target.value;
    });

    // Loading indicator
    function showLoading(elementId) {
        document.getElementById(elementId).innerHTML = '<div style="text-align:center;padding:40px;"><div class="loading"></div><p style="margin-top:20px;">Processing...</p></div>';
    }

    // Display results
    function displayResults(elementId, html) {
        document.getElementById(elementId).innerHTML = html;
    }

    // === Semantic Search ===
    function performSemanticSearch() {
        const query = document.getElementById('search-query').value;
        const documents = document.getElementById('search-documents').value.split('\\n').filter(d => d.trim());
        const topK = parseInt(document.getElementById('search-top-k').value);

        if (!query.trim()) {
            alert('Please enter a search query');
            return;
        }
        if (documents.length === 0) {
            alert('Please enter at least one document');
            return;
        }

        showLoading('search-results');
        Blink.msg('semantic_search', {
            query: query,
            documents: documents,
            topK: topK
        });
    }

    // === Text Similarity ===
    function compareSimilarity() {
        const text1 = document.getElementById('sim-text1').value;
        const text2 = document.getElementById('sim-text2').value;

        if (!text1.trim() || !text2.trim()) {
            alert('Please enter both texts');
            return;
        }

        showLoading('similarity-results');
        Blink.msg('compare_similarity', {
            text1: text1,
            text2: text2
        });
    }

    // === Classification ===
    function classifyText() {
        const text = document.getElementById('classify-text').value;
        const mode = document.getElementById('classify-mode').value;

        if (!text.trim()) {
            alert('Please enter text to classify');
            return;
        }

        showLoading('classification-results');
        Blink.msg('classify_text', {
            text: text,
            mode: mode
        });
    }

    // === Clustering ===
    function performClustering() {
        const texts = document.getElementById('cluster-texts').value.split('\\n').filter(t => t.trim());
        const threshold = parseFloat(document.getElementById('cluster-threshold').value);

        if (texts.length < 2) {
            alert('Please enter at least 2 texts to cluster');
            return;
        }

        showLoading('clustering-results');
        Blink.msg('cluster_texts', {
            texts: texts,
            threshold: threshold
        });
    }

    function findOutliers() {
        const texts = document.getElementById('cluster-texts').value.split('\\n').filter(t => t.trim());

        if (texts.length < 2) {
            alert('Please enter at least 2 texts');
            return;
        }

        showLoading('clustering-results');
        Blink.msg('find_outliers', {texts: texts});
    }

    // === Text Analysis ===
    function analyzeComplexity() {
        const text = document.getElementById('analysis-text').value;

        if (!text.trim()) {
            alert('Please enter text to analyze');
            return;
        }

        showLoading('analysis-results');
        Blink.msg('analyze_complexity', {text: text});
    }

    function analyzeCoherence() {
        const text = document.getElementById('analysis-text').value;

        if (!text.trim()) {
            alert('Please enter text to analyze');
            return;
        }

        showLoading('analysis-results');
        Blink.msg('analyze_coherence', {text: text});
    }

    function analyzeReadability() {
        const text = document.getElementById('analysis-text').value;

        if (!text.trim()) {
            alert('Please enter text to analyze');
            return;
        }

        showLoading('analysis-results');
        Blink.msg('analyze_readability', {text: text});
    }

    function analyzeTokens() {
        const text = document.getElementById('analysis-text').value;

        if (!text.trim()) {
            alert('Please enter text to analyze');
            return;
        }

        showLoading('analysis-results');
        Blink.msg('analyze_tokens', {text: text});
    }

    console.log('EngineT GUI loaded successfully');
    """
end

"""
Setup message handlers for communication with Julia backend
"""
function setup_handlers(w)
    # Semantic Search Handler
    handle(w, "semantic_search") do args
        query = args["query"]
        documents = args["documents"]
        top_k = args["topK"]

        @async try
            results = semantic_search(state.engine, query, documents, top_k=top_k)

            html = """<div class="result-card">
            <h4>🔍 Search Results ($(length(results)) found)</h4>"""

            for (i, result) in enumerate(results)
                score_pct = round(result.similarity * 100, digits=2)
                html *= """
                <div class="result-item">
                <span><strong>#$i:</strong> $(first(result.document, 150))$(length(result.document) > 150 ? "..." : "")</span>
                <span class="score">$score_pct%</span>
                </div>
                <div class="progress-bar">
                <div class="progress-fill" style="width: $score_pct%;">$score_pct%</div>
                </div>
                """
            end

            html *= "</div>"

            @js w displayResults("search-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("search-results", $error_html)
        end
    end

    # Similarity Comparison Handler
    handle(w, "compare_similarity") do args
        text1 = args["text1"]
        text2 = args["text2"]

        @async try
            sim = text_similarity(state.engine, text1, text2)
            sim_pct = round(sim * 100, digits=2)

            # Also calculate other similarity metrics
            euc_sim = euclidean_similarity(state.engine, text1, text2)
            euc_pct = round(euc_sim * 100, digits=2)

            html = """<div class="result-card">
            <h4>📊 Similarity Analysis</h4>
            <div class="result-item">
            <span>Cosine Similarity:</span>
            <span class="score">$sim_pct%</span>
            </div>
            <div class="progress-bar">
            <div class="progress-fill" style="width: $sim_pct%;">$sim_pct%</div>
            </div>
            <div class="result-item">
            <span>Euclidean Similarity:</span>
            <span class="score">$euc_pct%</span>
            </div>
            <div class="progress-bar">
            <div class="progress-fill" style="width: $euc_pct%;">$euc_pct%</div>
            </div>
            <div class="result-item">
            <span>Interpretation:</span>
            <span>$(sim > 0.8 ? "Very Similar" : sim > 0.6 ? "Similar" : sim > 0.4 ? "Somewhat Similar" : "Different")</span>
            </div>
            </div>"""

            @js w displayResults("similarity-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("similarity-results", $error_html)
        end
    end

    # Classification Handler
    handle(w, "classify_text") do args
        text = args["text"]
        mode = args["mode"]

        @async try
            result = if mode == "sentiment"
                classify_sentiment(state.engine, text)
            else
                detect_topics(state.engine, text)
            end

            html = """<div class="result-card">
            <h4>🏷️ Classification Results</h4>"""

            if mode == "sentiment"
                conf_pct = round(result.confidence * 100, digits=2)
                pos_pct = round(result.positive_score * 100, digits=2)
                neg_pct = round(result.negative_score * 100, digits=2)

                html *= """
                <div class="result-item">
                <span>Sentiment:</span>
                <span class="score">$(uppercase(result.sentiment))</span>
                </div>
                <div class="result-item">
                <span>Confidence:</span>
                <span class="score">$conf_pct%</span>
                </div>
                <div class="result-item">
                <span>Positive Score:</span>
                <span>$pos_pct%</span>
                </div>
                <div class="progress-bar">
                <div class="progress-fill" style="width: $pos_pct%; background: linear-gradient(90deg, #4caf50, #66bb6a);">$pos_pct%</div>
                </div>
                <div class="result-item">
                <span>Negative Score:</span>
                <span>$neg_pct%</span>
                </div>
                <div class="progress-bar">
                <div class="progress-fill" style="width: $neg_pct%; background: linear-gradient(90deg, #f44336, #e57373);">$neg_pct%</div>
                </div>
                """
            else  # topic detection
                conf_pct = round(result.confidence * 100, digits=2)
                html *= """
                <div class="result-item">
                <span>Primary Topic:</span>
                <span class="score">$(uppercase(result.primary_topic))</span>
                </div>
                <div class="result-item">
                <span>Confidence:</span>
                <span class="score">$conf_pct%</span>
                </div>
                <h4 style="margin-top: 15px;">All Topics:</h4>
                """
                for (topic, score) in sort(collect(result.all_topics), by=x -> x[2], rev=true)
                    score_pct = round(score * 100, digits=2)
                    html *= """
                    <div class="result-item">
                    <span>$(uppercase(topic)):</span>
                    <span>$score_pct%</span>
                    </div>
                    <div class="progress-bar">
                    <div class="progress-fill" style="width: $score_pct%;">$score_pct%</div>
                    </div>
                    """
                end
            end

            html *= "</div>"

            @js w displayResults("classification-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("classification-results", $error_html)
        end
    end

    # Clustering Handler
    handle(w, "cluster_texts") do args
        texts = args["texts"]
        threshold = args["threshold"]

        @async try
            clusters = cluster_texts(state.engine, texts, similarity_threshold=threshold)

            html = """<div class="result-card">
            <h4>🗂️ Clustering Results ($(length(clusters)) clusters found)</h4>"""

            for (i, cluster) in enumerate(clusters)
                html *= """
                <div class="result-card" style="margin: 15px 0;">
                <h4>Cluster $i ($(cluster.size) texts)</h4>
                """
                for (j, text) in enumerate(cluster.texts)
                    html *= """<div class="result-item">
                    <span><strong>Text $j:</strong> $(first(text, 100))$(length(text) > 100 ? "..." : "")</span>
                    </div>"""
                end
                html *= "</div>"
            end

            html *= "</div>"

            @js w displayResults("clustering-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("clustering-results", $error_html)
        end
    end

    # Outliers Handler
    handle(w, "find_outliers") do args
        texts = args["texts"]

        @async try
            outliers = find_outliers(state.engine, texts)

            html = """<div class="result-card">
            <h4>🎯 Outlier Detection ($(length(outliers)) outliers found)</h4>"""

            if length(outliers) == 0
                html *= """<p>No outliers detected. All texts are similar to each other.</p>"""
            else
                for outlier in outliers
                    sim_pct = round(outlier.avg_similarity * 100, digits=2)
                    html *= """
                    <div class="result-item">
                    <span><strong>Text $(outlier.index):</strong> $(first(outlier.text, 120))$(length(outlier.text) > 120 ? "..." : "")</span>
                    <span class="score">Avg Sim: $sim_pct%</span>
                    </div>
                    <div class="progress-bar">
                    <div class="progress-fill" style="width: $sim_pct%; background: linear-gradient(90deg, #ff9800, #ffb74d);">$sim_pct%</div>
                    </div>
                    """
                end
            end

            html *= "</div>"

            @js w displayResults("clustering-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("clustering-results", $error_html)
        end
    end

    # Complexity Analysis Handler
    handle(w, "analyze_complexity") do args
        text = args["text"]

        @async try
            result = analyze_text_complexity(state.engine, text)

            comp_pct = round(result.complexity_score * 100, digits=2)
            simple_pct = round(result.simple_similarity * 100, digits=2)
            complex_pct = round(result.complex_similarity * 100, digits=2)

            html = """<div class="result-card">
            <h4>📈 Complexity Analysis</h4>
            <div class="result-item">
            <span>Assessment:</span>
            <span class="score">$(uppercase(result.assessment))</span>
            </div>
            <div class="result-item">
            <span>Complexity Score:</span>
            <span class="score">$comp_pct%</span>
            </div>
            <div class="progress-bar">
            <div class="progress-fill" style="width: $comp_pct%;">$comp_pct%</div>
            </div>
            <div class="result-item">
            <span>Similarity to Simple Text:</span>
            <span>$simple_pct%</span>
            </div>
            <div class="result-item">
            <span>Similarity to Complex Text:</span>
            <span>$complex_pct%</span>
            </div>
            </div>"""

            @js w displayResults("analysis-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("analysis-results", $error_html)
        end
    end

    # Coherence Analysis Handler
    handle(w, "analyze_coherence") do args
        text = args["text"]

        @async try
            result = measure_coherence(state.engine, text)

            coh_pct = round(result.coherence_score * 100, digits=2)

            html = """<div class="result-card">
            <h4>🔗 Coherence Analysis</h4>
            <div class="result-item">
            <span>Assessment:</span>
            <span class="score">$(uppercase(result.assessment))</span>
            </div>
            <div class="result-item">
            <span>Coherence Score:</span>
            <span class="score">$coh_pct%</span>
            </div>
            <div class="progress-bar">
            <div class="progress-fill" style="width: $coh_pct%;">$coh_pct%</div>
            </div>
            <div class="result-item">
            <span>Number of Sentences:</span>
            <span>$(result.sentence_count)</span>
            </div>
            </div>"""

            @js w displayResults("analysis-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("analysis-results", $error_html)
        end
    end

    # Readability Analysis Handler
    handle(w, "analyze_readability") do args
        text = args["text"]

        @async try
            result = analyze_readability(state.engine, text)

            read_pct = round(result.readability_score * 100, digits=2)
            simple_pct = round(result.simple_similarity * 100, digits=2)
            complex_pct = round(result.complex_similarity * 100, digits=2)

            html = """<div class="result-card">
            <h4>📖 Readability Analysis</h4>
            <div class="result-item">
            <span>Assessment:</span>
            <span class="score">$(uppercase(result.assessment))</span>
            </div>
            <div class="result-item">
            <span>Readability Score:</span>
            <span class="score">$read_pct%</span>
            </div>
            <div class="progress-bar">
            <div class="progress-fill" style="width: $read_pct%;">$read_pct%</div>
            </div>
            <div class="result-item">
            <span>Simple Similarity:</span>
            <span>$simple_pct%</span>
            </div>
            <div class="result-item">
            <span>Complex Similarity:</span>
            <span>$complex_pct%</span>
            </div>
            </div>"""

            @js w displayResults("analysis-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("analysis-results", $error_html)
        end
    end

    # Token Analysis Handler
    handle(w, "analyze_tokens") do args
        text = args["text"]

        @async try
            analysis = analyze_context(state.engine, text)

            html = """<div class="result-card">
            <h4>🔤 Token Analysis</h4>
            <div class="result-item">
            <span>Token Count:</span>
            <span class="score">$(analysis["token_count"])</span>
            </div>
            <div class="result-item">
            <span>Embedding Dimension:</span>
            <span class="score">$(analysis["embedding_dim"])</span>
            </div>
            <div class="result-item">
            <span>Vocabulary Coverage:</span>
            <span class="score">$(round(analysis["vocab_coverage"] * 100, digits=2))%</span>
            </div>
            <div class="result-item">
            <span>Decoded Text:</span>
            <span style="font-family: monospace;">$(analysis["decoded_text"])</span>
            </div>
            <div class="result-item">
            <span>Token IDs:</span>
            <span style="font-family: monospace;">$(join(analysis["token_ids"][1:min(15, length(analysis["token_ids"]))], ", "))$(length(analysis["token_ids"]) > 15 ? "..." : "")</span>
            </div>
            </div>"""

            @js w displayResults("analysis-results", $html)
        catch e
            error_html = """<div class="result-card" style="border-left-color: var(--error);">
            <h4>❌ Error</h4>
            <p>$(string(e))</p>
            </div>"""
            @js w displayResults("analysis-results", $error_html)
        end
    end

    # Theme toggle handler
    handle(w, "toggle_theme") do args
        state.current_theme = state.current_theme == "dark" ? "light" : "dark"
        println("Theme switched to: $(state.current_theme)")
    end
end

end # module
