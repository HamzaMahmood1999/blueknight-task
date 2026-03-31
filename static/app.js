document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    const modeToggle = document.getElementById('mode-toggle');
    const labelRaw = document.getElementById('label-raw');
    const labelAgent = document.getElementById('label-agent');
    
    const agentConsole = document.getElementById('agent-console');
    const logStream = document.getElementById('log-stream');
    const finalRationale = document.getElementById('final-rationale');
    
    const resultsHeader = document.getElementById('results-header');
    const resultsCount = document.getElementById('results-count');
    const queryBadges = document.getElementById('query-badges');
    const resultsGrid = document.getElementById('results-grid');

    // Modal elements
    const modal = document.getElementById('company-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const modalName = document.getElementById('modal-company-name');
    const modalCountry = document.getElementById('modal-company-country');
    const modalBio = document.getElementById('modal-company-bio');

    let currentResults = [];

    // Modal Close Logic
    closeModalBtn.addEventListener('click', () => modal.classList.add('hidden'));
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.add('hidden');
    });

    // Card Click Delegate
    resultsGrid.addEventListener('click', (e) => {
        const card = e.target.closest('.company-card');
        if (card) {
            const index = card.getAttribute('data-index');
            const company = currentResults[index];
            if (company) {
                modalName.textContent = company.company_name;
                modalCountry.textContent = company.country || 'Global';
                modalBio.textContent = company.long_offering;
                modal.classList.remove('hidden');
            }
        }
    });

    // Toggle Mode Logic
    modeToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            labelAgent.classList.add('active');
            labelRaw.classList.remove('active');
        } else {
            labelRaw.classList.add('active');
            labelAgent.classList.remove('active');
        }
    });

    const setProcessingState = (isProcessing) => {
        searchBtn.disabled = isProcessing;
        searchInput.disabled = isProcessing;
        if (isProcessing) {
            btnText.classList.add('hidden');
            btnLoader.classList.remove('hidden');
            resultsGrid.innerHTML = '';
            resultsHeader.classList.add('hidden');
            logStream.innerHTML = '';
            finalRationale.classList.add('hidden');
            finalRationale.textContent = '';
            if (modeToggle.checked) {
                agentConsole.classList.remove('hidden');
            } else {
                agentConsole.classList.add('hidden');
            }
        } else {
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            document.querySelector('.pulse')?.classList.remove('pulse'); // stop pulsing target
        }
    };

    const addLog = (msg, type = "normal") => {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;
        div.textContent = `> ${msg}`;
        logStream.appendChild(div);
        logStream.scrollTop = logStream.scrollHeight;
    };

    const renderBadges = (queryPayload) => {
        queryBadges.innerHTML = '';
        
        // Query Badge
        if (queryPayload.query_text) {
            queryBadges.innerHTML += `<span class="badge badge-query">🔍 ${queryPayload.query_text}</span>`;
        }
        
        // Geo Badges
        if (queryPayload.geography && queryPayload.geography.length > 0) {
            queryPayload.geography.forEach(geo => {
                queryBadges.innerHTML += `<span class="badge badge-geo">📍 ${geo}</span>`;
            });
        }
        
        // Exclusions Badges
        if (queryPayload.exclusions && queryPayload.exclusions.length > 0) {
            queryPayload.exclusions.forEach(ex => {
                queryBadges.innerHTML += `<span class="badge badge-exclude">🚫 ${ex}</span>`;
            });
        }
    };

    const renderResults = (data, queryPayload) => {
        resultsCount.textContent = `${data.total || data.results.length} Matches Found`;
        renderBadges(queryPayload);
        resultsHeader.classList.remove('hidden');
        currentResults = data.results;

        if (data.results.length === 0) {
            resultsGrid.innerHTML = `<p style="color: #94a3b8; grid-column: 1/-1;">No companies met the criteria.</p>`;
            return;
        }

        resultsGrid.innerHTML = data.results.map((r, i) => `
            <div class="company-card" data-index="${i}">
                <div class="card-header">
                    <div>
                        <div class="company-name">${r.company_name}</div>
                        <div class="company-country">${r.country || 'Global'}</div>
                    </div>
                    <div class="score-badge">${r.score.toFixed(2)}</div>
                </div>
                <div class="company-bio">${r.long_offering.substring(0, 180)}...</div>
                <div class="score-breakdown">
                    <span class="score-pill">Vec: ${r.score_components.vector || 0}</span>
                    <span class="score-pill">Geo: ${r.score_components.geo_boost || 0}</span>
                    <span class="score-pill">Key: ${r.score_components.keyword_boost || 0}</span>
                </div>
            </div>
        `).join('');
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userInput = searchInput.value.trim();
        if (!userInput) return;

        const isAgentMode = modeToggle.checked;
        setProcessingState(true);

        try {
            if (isAgentMode) {
                addLog("Initiating Refinement Agent Loop...");
                const response = await fetch('/agent/refine', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userInput, max_iterations: 3 })
                });

                if (!response.ok) throw new Error("Agent request failed");
                const agentData = await response.json();
                
                // Show iterations logic purely from returned metadata
                const iterations = agentData.meta.iteration_details || [];
                iterations.forEach(it => {
                    addLog(`Iteration ${it.iteration}: Generated query -> "${it.query_text}"`);
                    addLog(`Retrieval stats: ${it.total} raw, ${it.filtered_count} discarded. Mean top-10 score: ${it.mean_score.toFixed(2)}`);
                });

                addLog(`Loop Terminated. Fetching final results grid...`, 'success');
                
                finalRationale.textContent = agentData.rationale;
                finalRationale.classList.remove('hidden');

                // Now execute the actual search using the refined query to get the rich company cards
                const searchReq = await fetch('/search/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: agentData.refined_query, 
                        top_k_raw: 1000, 
                        top_k_final: 20 
                    })
                });
                
                const searchData = await searchReq.json();
                renderResults(searchData, agentData.refined_query);

            } else {
                // Raw Search
                const rawQuery = {
                    query_text: userInput,
                    geography: [],
                    exclusions: []
                };
                
                const response = await fetch('/search/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: rawQuery, 
                        top_k_raw: 1000, 
                        top_k_final: 20 
                    })
                });

                if (!response.ok) throw new Error("Search request failed");
                const searchData = await response.json();
                renderResults(searchData, rawQuery);
            }
        } catch (error) {
            console.error(error);
            if (isAgentMode) addLog(`Error: ${error.message}`, 'error');
            alert("An error occurred. Check browser console.");
        } finally {
            setProcessingState(false);
            // restore pulse
            document.querySelector('.agent-console h3').innerHTML = '<span class="pulse"></span> Agent Reasoning Log';
        }
    });
});
