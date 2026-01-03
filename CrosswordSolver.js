// CrosswordSolver.js

import { predefinedPuzzles } from './puzzles.js';

export class CrosswordSolver {
    constructor() {
        // --------------------- Configuration & Flags ---------------------
        this.DEBUG = false;
        this.isSolving = false;
        this.isNumberEntryMode = false;
        this.isLetterEntryMode = false;
        this.isDragMode = false;
        this.isDragging = false;

        // --------------------- Data Structures ---------------------
        this.grid = [];
        this.words = [];
        this.wordLengthCache = {};
        this.letterFrequencies = {};
        this.slots = {};
        this.constraints = {};
        this.solution = {};
        this.domains = {};
        this.cellContents = {};
        this.cells = {};
        this.performanceData = {};
        this.recursiveCalls = 0;

        // --------------------- Predefined Puzzles ---------------------
        this.predefinedPuzzles = predefinedPuzzles;

        // --------------------- Drag Mode Variables ---------------------
        this.toggleToBlack = true;
        this.startDragBound = null;
        this.onDragBound = null;
        this.stopDragBound = null;

        // --------------------- WordNet Dictionary ---------------------
        this.dictionary = null; // set via setDictionary(dictionaryObj) after loading

        // --------------------- Method Binding ---------------------
        this.loadWords = this.loadWords.bind(this);
        this.generateGrid = this.generateGrid.bind(this);
        this.renderGrid = this.renderGrid.bind(this);
        this.loadPredefinedPuzzle = this.loadPredefinedPuzzle.bind(this);
        this.cellClicked = this.cellClicked.bind(this);
        this.startNumberEntryMode = this.startNumberEntryMode.bind(this);
        this.startLetterEntryMode = this.startLetterEntryMode.bind(this);
        this.startDragMode = this.startDragMode.bind(this);
        this.solveCrossword = this.solveCrossword.bind(this);
        this.handleSearchInput = this.handleSearchInput.bind(this);
        this.displaySearchResults = this.displaySearchResults.bind(this);
        this.showDefinitionPopup = this.showDefinitionPopup.bind(this);
        this.autoNumberGrid = this.autoNumberGrid.bind(this);

        // A simple cache for fallback definitions so we don't re-fetch the same word repeatedly
        this.fallbackCache = {};
    }

    // ----------------------------------------------------------------
    //                  Dictionary & Definition Popup
    // ----------------------------------------------------------------

    setDictionary(dictionaryObj) {
        this.dictionary = dictionaryObj;
        this.debugLog("Dictionary set. Example 'run':", this.dictionary['run']);
    }

    /**
     * showDefinitionPopup(rawWord):
     * 1) Look up rawWord in our local WordNet dictionary.
     * 2) If found, display the definitions from "WordNet".
     * 3) If not found or empty, fetch from a fallback dictionary API.
     *    - If successful, show from "DictionaryAPI" or whichever source.
     *    - If still not found, show "No definition found."
     */
    async showDefinitionPopup(rawWord) {
        if (!rawWord) return;
        const word = rawWord.toLowerCase();

        // 1) Attempt local dictionary first
        const senses = this.dictionary ? this.dictionary[word] : null;
        if (senses && senses.length > 0) {
            // We have at least one sense from WordNet
            this.displayDefinitionsPopup(rawWord, senses, "WordNet");
            return;
        }

        // 2) Fallback: fetch from a dictionary API
        try {
            const fallbackData = await this.fetchFallbackDefinition(word);
            if (fallbackData && fallbackData.length > 0) {
                // Convert fallbackData to the same shape we use for display
                // e.g. an array of { pos, definitions: [...] }
                const senses = this.transformFallbackData(fallbackData);
                if (senses.length > 0) {
                    this.displayDefinitionsPopup(rawWord, senses, "DictionaryAPI");
                    return;
                }
            }

            // 3) If still no definition, display "No definition found."
            this.createPopup(`
              <h2>${rawWord}</h2>
              <p><em>No definition found.</em></p>
              <small>Source: DictionaryAPI</small>
            `);
        } catch (err) {
            console.warn("Fallback definition fetch failed:", err);
            // Show a basic "no definition" message
            this.createPopup(`
              <h2>${rawWord}</h2>
              <p><em>No definition found (API error).</em></p>
              <small>Source: DictionaryAPI</small>
            `);
        }
    }

    /**
     * fetchFallbackDefinition(word):
     * Uses a free dictionary API (e.g. https://api.dictionaryapi.dev/api/v2/entries/en/<word>)
     * Returns the JSON data or null if not found.
     */
    async fetchFallbackDefinition(word) {
        // If we already have it in our fallbackCache, return it
        if (this.fallbackCache[word]) {
            return this.fallbackCache[word];
        }

        const url = `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`;
        const resp = await fetch(url);
        if (!resp.ok) {
            // e.g. 404 "No definitions found" or other error
            throw new Error(`API returned status ${resp.status}`);
        }
        const data = await resp.json();
        // Cache it
        this.fallbackCache[word] = data;
        return data;
    }

    /**
     * transformFallbackData(data):
     * Transforms the dictionaryapi.dev JSON shape into an array of:
     *    [ { pos: 'n', definitions: [ 'Definition text', ... ]}, ... ]
     */
    transformFallbackData(data) {
        // data typically looks like:
        // [
        //   {
        //     "word": "hello",
        //     "phonetics": [...],
        //     "meanings": [
        //       {
        //         "partOfSpeech": "interjection",
        //         "definitions": [
        //           { "definition": "A greeting", "example": "Hello there!" }
        //         ]
        //       },
        //       ...
        //     ]
        //   }
        // ]
        // We'll flatten them
        const result = [];
        if (!Array.isArray(data)) return result;

        for (const entry of data) {
            if (!entry.meanings) continue;
            for (const meaning of entry.meanings) {
                const pos = meaning.partOfSpeech || 'unknown';
                const defList = (meaning.definitions || []).map(d => d.definition || '');
                if (defList.length > 0) {
                    result.push({ pos, definitions: defList });
                }
            }
        }
        return result;
    }

    /**
     * displayDefinitionsPopup(rawWord, senses, sourceLabel):
     * Renders the definitions in a popup for the user.
     */
    displayDefinitionsPopup(rawWord, senses, sourceLabel) {
        let innerHTML = `<h2>${rawWord}</h2>`;

        // Group senses by part-of-speech
        const sensesByPos = {};
        for (const sense of senses) {
            const posKey = sense.pos || 'unknown';
            if (!sensesByPos[posKey]) {
                sensesByPos[posKey] = [];
            }
            // sense.definitions is an array of strings
            sensesByPos[posKey].push(sense.definitions);
        }

        // Output each POS + definitions
        Object.keys(sensesByPos).forEach(posKey => {
            const definitionsForPos = sensesByPos[posKey];
            const flattenedDefs = definitionsForPos.flat();
            innerHTML += `<h4>(${posKey})</h4>`;
            flattenedDefs.forEach((def, idx) => {
                innerHTML += `<p>${idx + 1}. ${def}</p>`;
            });
        });

        innerHTML += `<small>Source: ${sourceLabel}</small>`;
        this.createPopup(innerHTML);
    }

    /**
     * Basic popup creation
     */
    createPopup(contentHTML) {
        const overlay = document.createElement('div');
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100vw';
        overlay.style.height = '100vh';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.zIndex = '9998';
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
            }
        });

        const popup = document.createElement('div');
        popup.style.position = 'fixed';
        popup.style.top = '50%';
        popup.style.left = '50%';
        popup.style.transform = 'translate(-50%, -50%)';
        popup.style.width = '600px';
        popup.style.maxHeight = '70vh';
        popup.style.backgroundColor = '#fff';
        popup.style.borderRadius = '4px';
        popup.style.padding = '20px';
        popup.style.overflowY = 'auto';
        popup.style.zIndex = '9999';

        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const titleBar = document.createElement('span');
        titleBar.textContent = 'Definition Lookup';

        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'X';
        closeBtn.style.marginLeft = '10px';
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(overlay);
        });

        header.appendChild(titleBar);
        header.appendChild(closeBtn);

        const contentDiv = document.createElement('div');
        contentDiv.innerHTML = contentHTML;

        popup.appendChild(header);
        popup.appendChild(contentDiv);
        overlay.appendChild(popup);

        document.body.appendChild(overlay);
    }

    // ----------------------------------------------------------------
    //                          Initialization
    // ----------------------------------------------------------------

    init() {
        try {
            this.createEventListeners();
            // Load from Words.txt (legacy approach)
            this.loadWords()
                .then(() => this.generateGrid(10, 10))
                .catch((error) => {
                    this.handleError("Initialization failed during word loading.", error);
                });
        } catch (error) {
            this.handleError("Initialization failed.", error);
        }
    }

    createEventListeners() {
        try {
            this.bindButton('#generate-grid-button', () => {
                const rowsInput = document.getElementById('rows-input');
                const columnsInput = document.getElementById('columns-input');
                const rows = parseInt(rowsInput?.value) || 10;
                const cols = parseInt(columnsInput?.value) || 10;
                this.generateGrid(rows, cols);
            });

            this.bindButton('#load-easy-button', () => this.loadPredefinedPuzzle("Easy"));
            this.bindButton('#load-medium-button', () => this.loadPredefinedPuzzle("Medium"));
            this.bindButton('#load-hard-button', () => this.loadPredefinedPuzzle("Hard"));

            this.bindButton('#number-entry-button', this.startNumberEntryMode);
            // Ensure you do the same for the new Auto-Number button:
            this.bindButton('#auto-number-button', this.autoNumberGrid);

            this.bindButton('#letter-entry-button', this.startLetterEntryMode);
            this.bindButton('#drag-mode-button', this.startDragMode);

            this.bindButton('#solve-crossword-button', this.solveCrossword);

            const wordSearchInput = document.getElementById('word-search-input');
            if (wordSearchInput) {
                wordSearchInput.addEventListener('input', this.handleSearchInput);
            }
        } catch (error) {
            this.handleError("Error setting up event listeners:", error);
        }
    }

    bindButton(selector, callback) {
        const button = document.querySelector(selector);
        if (!button) {
            this.debugLog(`Button with selector "${selector}" not found.`);
            return;
        }
        button.addEventListener('click', callback);
    }

    // ----------------------------------------------------------------
    //                          Word Loading
    // ----------------------------------------------------------------

    async loadWords() {
        try {
            const response = await fetch('Data/Words.txt');
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }
            const text = await response.text();
            const rawWords = text
                .split(/\r?\n/)
                .map(w => w.trim().toUpperCase())
                .filter(Boolean);

            if (!rawWords.every(word => /^[A-Z]+$/.test(word))) {
                throw new Error("File contains invalid words. Ensure all entries are purely alphabetic.");
            }

            this.words = rawWords;
            this.cacheWordsByLength();
            this.calculateLetterFrequencies();
            this.debugLog(`Words loaded: ${this.words.length}`);
            this.updateStatus(`Words loaded successfully: ${this.words.length}`);

        } catch (error) {
            this.words = ["LASER", "SAILS", "SHEET", "STEER", "HEEL", "HIKE", "KEEL", "KNOT"];
            alert("Warning: Words.txt not found or invalid. Using fallback word list.");
            this.debugLog("Words.txt not found or invalid. Using fallback word list.");
            this.updateStatus("Warning: Words.txt not found or invalid. Using fallback word list.", true);
            this.cacheWordsByLength();
            this.calculateLetterFrequencies();
        }
    }

    cacheWordsByLength() {
        this.wordLengthCache = {};
        for (const word of this.words) {
            const len = word.length;
            if (!this.wordLengthCache[len]) {
                this.wordLengthCache[len] = [];
            }
            this.wordLengthCache[len].push(word);
        }
        this.debugLog("Word length cache created.");
    }

    calculateLetterFrequencies() {
        this.letterFrequencies = {};
        for (const word of this.words) {
            for (const char of word) {
                if (!this.letterFrequencies[char]) {
                    this.letterFrequencies[char] = 0;
                }
                this.letterFrequencies[char]++;
            }
        }
    }

    // ----------------------------------------------------------------
    //                       Auto-Numbering
    // ----------------------------------------------------------------

    autoNumberGrid() {
        try {
            // Wipe existing numbers
            for (let r = 0; r < this.grid.length; r++) {
                for (let c = 0; c < this.grid[0].length; c++) {
                    if (/\d+/.test(this.grid[r][c])) {
                        this.grid[r][c] = ' ';
                    }
                }
            }

            let currentNumber = 1;
            for (let r = 0; r < this.grid.length; r++) {
                for (let c = 0; c < this.grid[0].length; c++) {
                    if (this.isStartOfAcrossSlot(r, c)) {
                        this.grid[r][c] = currentNumber.toString();
                        currentNumber++;
                    }
                    if (this.isStartOfDownSlot(r, c)) {
                        if (!/\d+/.test(this.grid[r][c])) {
                            this.grid[r][c] = currentNumber.toString();
                            currentNumber++;
                        }
                    }
                }
            }

            this.renderGrid();
            this.updateStatus("Auto-numbered all slots.");

        } catch (error) {
            this.handleError("Error auto-numbering grid:", error);
        }
    }

    isStartOfAcrossSlot(r, c) {
        if (!this.grid[r] || !this.grid[r][c]) return false;
        if (this.grid[r][c] === '#') return false;
        if (c === 0 || this.grid[r][c - 1] === '#') {
            if (c + 1 < this.grid[0].length && this.grid[r][c + 1] !== '#') {
                return true;
            }
        }
        return false;
    }

    isStartOfDownSlot(r, c) {
        if (!this.grid[r] || !this.grid[r][c]) return false;
        if (this.grid[r][c] === '#') return false;
        if (r === 0 || this.grid[r - 1][c] === '#') {
            if (r + 1 < this.grid.length && this.grid[r + 1][c] !== '#') {
                return true;
            }
        }
        return false;
    }

    // ----------------------------------------------------------------
    //                       Grid Management
    // ----------------------------------------------------------------

    generateGrid(rows = 10, cols = 10) {
        try {
            this.grid = Array.from({ length: rows }, () => Array(cols).fill("#"));
            this.resetDataStructures();
            this.renderGrid();
            this.updateStatus(`Grid generated with rows: ${rows}, columns: ${cols}`, true);
        } catch (error) {
            this.handleError("Error generating grid:", error);
        }
    }

    resetDataStructures() {
        this.solution = {};
        this.slots = {};
        this.constraints = {};
        this.domains = {};
        this.cellContents = {};
        this.cells = {};
    }

    renderGrid() {
        try {
            const gridContainer = document.getElementById('grid-container');
            if (!gridContainer) {
                throw new Error("Grid container not found in DOM.");
            }
            gridContainer.innerHTML = '';

            const table = document.createElement('table');
            table.style.borderCollapse = 'collapse';

            for (let r = 0; r < this.grid.length; r++) {
                const tr = document.createElement('tr');
                for (let c = 0; c < this.grid[0].length; c++) {
                    const td = document.createElement('td');
                    td.style.border = '1px solid #ccc';
                    td.style.width = '30px';
                    td.style.height = '30px';
                    td.style.textAlign = 'center';
                    td.style.verticalAlign = 'middle';
                    td.style.fontSize = '16px';
                    td.style.fontWeight = 'bold';
                    td.style.cursor = 'pointer';
                    td.dataset.row = r;
                    td.dataset.col = c;

                    if (this.grid[r][c] === "#") {
                        td.style.backgroundColor = '#000';
                        td.style.color = '#000';
                    } else {
                        td.style.backgroundColor = '#fff';
                        td.style.color = '#444';
                        if (/\d/.test(this.grid[r][c])) {
                            td.textContent = this.grid[r][c];
                        }
                    }

                    td.addEventListener('click', this.cellClicked);
                    this.cells[`${r},${c}`] = td;
                    tr.appendChild(td);
                }
                table.appendChild(tr);
            }

            gridContainer.appendChild(table);

        } catch (error) {
            this.handleError("Error rendering grid:", error);
        }
    }

    loadPredefinedPuzzle(puzzleName) {
        try {
            const puzzle = this.predefinedPuzzles.find(p => p.name === puzzleName);
            if (!puzzle) {
                throw new Error(`Puzzle "${puzzleName}" not found.`);
            }
            if (!Array.isArray(puzzle.grid) || puzzle.grid.length === 0 || !Array.isArray(puzzle.grid[0])) {
                throw new Error(`Invalid grid format for puzzle "${puzzleName}".`);
            }

            const rows = puzzle.grid.length;
            const cols = puzzle.grid[0].length;

            this.grid = [];
            this.resetDataStructures();

            this.grid = puzzle.grid.map(row => [...row]);
            this.renderGrid();

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const cellValue = this.grid[r][c];
                    const key = `${r},${c}`;
                    if (/[A-Z]/.test(cellValue)) {
                        this.cellContents[key] = cellValue;
                    } else if (cellValue !== "#" && cellValue.trim() !== "") {
                        this.cellContents[key] = null;
                    }
                }
            }

            this.updateStatus(`Loaded predefined puzzle: ${puzzleName}`, true);

        } catch (error) {
            this.handleError(`Error loading puzzle "${puzzleName}":`, error);
        }
    }

    cellClicked(event) {
        try {
            const cell = event.target;
            const row = parseInt(cell.dataset.row, 10);
            const col = parseInt(cell.dataset.col, 10);

            if (this.isNumberEntryMode) {
                if (/\d/.test(this.grid[row][col])) {
                    this.removeNumberFromCell(row, col);
                } else {
                    this.addNumberToCell(row, col);
                }
                return;
            }

            if (this.isLetterEntryMode) {
                const letter = prompt("Enter a single letter (A-Z):");
                if (letter && /^[A-Za-z]$/.test(letter)) {
                    const upperLetter = letter.toUpperCase();
                    this.updateCell(row, col, upperLetter, '#000', '#fff');
                    this.grid[row][col] = upperLetter;
                } else if (letter !== null) {
                    alert("Please enter a single letter (A-Z).");
                }
                return;
            }

            if (this.isDragMode) {
                return;
            }

            if (this.grid[row][col] !== "#") {
                this.updateCell(row, col, '', '#000', '#000');
                this.grid[row][col] = "#";
                this.updateNumbersAfterRemoval(row, col);
            } else {
                this.updateCell(row, col, '', '#444', '#fff');
                this.grid[row][col] = " ";
            }
        } catch (error) {
            this.handleError("Error handling cell click:", error);
        }
    }

    // ----------------------------------------------------------------
    //                         Mode Toggles
    // ----------------------------------------------------------------

    startNumberEntryMode() {
        try {
            const modeLabel = document.getElementById('mode-label');
            const button = document.getElementById('number-entry-button');
            if (!modeLabel || !button) throw new Error("Number Entry mode elements not found.");

            if (this.isNumberEntryMode) {
                this.isNumberEntryMode = false;
                modeLabel.textContent = "Mode: Default";
                this.updateStatus("Number Entry Mode Deactivated.");
                button.textContent = "Number Entry Mode";
                button.style.backgroundColor = "#0069d9";
            } else {
                if (this.isLetterEntryMode) this.startLetterEntryMode();
                if (this.isDragMode) this.startDragMode();

                this.isNumberEntryMode = true;
                modeLabel.textContent = "Mode: Number Entry";
                this.updateStatus("Number Entry Mode Activated.");
                button.textContent = "Exit Number Entry Mode";
                button.style.backgroundColor = "#dc3545";
            }
        } catch (error) {
            this.handleError("Error toggling Number Entry Mode:", error);
        }
    }

    startLetterEntryMode() {
        try {
            const modeLabel = document.getElementById('mode-label');
            const button = document.getElementById('letter-entry-button');
            if (!modeLabel || !button) throw new Error("Letter Entry mode elements not found.");

            if (this.isLetterEntryMode) {
                this.isLetterEntryMode = false;
                modeLabel.textContent = "Mode: Default";
                this.updateStatus("Letter Entry Mode Deactivated.");
                button.textContent = "Letter Entry Mode";
                button.style.backgroundColor = "#0069d9";
            } else {
                if (this.isNumberEntryMode) this.startNumberEntryMode();
                if (this.isDragMode) this.startDragMode();

                this.isLetterEntryMode = true;
                modeLabel.textContent = "Mode: Letter Entry";
                this.updateStatus("Letter Entry Mode Activated.");
                button.textContent = "Exit Letter Entry Mode";
                button.style.backgroundColor = "#dc3545";
            }
        } catch (error) {
            this.handleError("Error toggling Letter Entry Mode:", error);
        }
    }

    startDragMode() {
        try {
            const modeLabel = document.getElementById('mode-label');
            const button = document.getElementById('drag-mode-button');
            if (!modeLabel || !button) throw new Error("Drag Mode elements not found.");

            if (this.isDragMode) {
                this.isDragMode = false;
                modeLabel.textContent = "Mode: Default";
                this.updateStatus("Drag Mode Deactivated.");
                button.textContent = "Drag Mode";
                button.style.backgroundColor = "#0069d9";

                for (const key in this.cells) {
                    const cell = this.cells[key];
                    cell.removeEventListener('mousedown', this.startDragBound);
                    cell.removeEventListener('mousemove', this.onDragBound);
                    cell.removeEventListener('mouseup', this.stopDragBound);
                }
            } else {
                if (this.isNumberEntryMode) this.startNumberEntryMode();
                if (this.isLetterEntryMode) this.startLetterEntryMode();

                this.isDragMode = true;
                modeLabel.textContent = "Mode: Drag";
                this.updateStatus("Drag Mode Activated.");
                button.textContent = "Exit Drag Mode";
                button.style.backgroundColor = "#dc3545";

                this.startDragBound = this.startDrag.bind(this);
                this.onDragBound = this.onDrag.bind(this);
                this.stopDragBound = this.stopDrag.bind(this);

                for (const key in this.cells) {
                    const cell = this.cells[key];
                    cell.addEventListener('mousedown', this.startDragBound);
                    cell.addEventListener('mousemove', this.onDragBound);
                    cell.addEventListener('mouseup', this.stopDragBound);
                }
            }
        } catch (error) {
            this.handleError("Error toggling Drag Mode:", error);
        }
    }

    // ----------------------------------------------------------------
    //                        Number Management
    // ----------------------------------------------------------------

    addNumberToCell(row, col) {
        try {
            const numberPositions = this.getNumberPositions();
            const newNumber = this.getNewNumber(row, col, numberPositions);
            this.updateNumbersAfterInsertion(row, col, newNumber);
            this.updateCell(row, col, newNumber.toString(), '#000', '#fff');
            this.grid[row][col] = newNumber.toString();
        } catch (error) {
            this.handleError(`Error adding number to cell (${row}, ${col}):`, error);
        }
    }

    getNumberPositions() {
        const numberPositions = [];
        for (let r = 0; r < this.grid.length; r++) {
            for (let c = 0; c < this.grid[0].length; c++) {
                if (/\d/.test(this.grid[r][c])) {
                    numberPositions.push({ number: parseInt(this.grid[r][c], 10), row: r, col: c });
                }
            }
        }
        numberPositions.sort((a, b) => a.number - b.number);
        return numberPositions;
    }

    getNewNumber(row, col, numberPositions) {
        let position = 0;
        for (let i = 0; i < numberPositions.length; i++) {
            const pos = numberPositions[i];
            if (row < pos.row || (row === pos.row && col < pos.col)) {
                break;
            }
            position = i + 1;
        }
        return position + 1;
    }

    updateNumbersAfterInsertion(row, col, newNumber) {
        for (let r = 0; r < this.grid.length; r++) {
            for (let c = 0; c < this.grid[0].length; c++) {
                const cellValue = this.grid[r][c];
                if (/\d/.test(cellValue)) {
                    const current = parseInt(cellValue, 10);
                    if (current >= newNumber && (r !== row || c !== col)) {
                        const updated = current + 1;
                        this.grid[r][c] = updated.toString();
                        this.updateCell(r, c, updated.toString());
                    }
                }
            }
        }
    }

    removeNumberFromCell(row, col) {
        try {
            const removed = parseInt(this.grid[row][col], 10);
            this.grid[row][col] = " ";
            this.updateCell(row, col, "", '#444', '#fff');

            for (let r = 0; r < this.grid.length; r++) {
                for (let c = 0; c < this.grid[0].length; c++) {
                    const cellValue = this.grid[r][c];
                    if (/\d/.test(cellValue)) {
                        const current = parseInt(cellValue, 10);
                        if (current > removed) {
                            const updated = current - 1;
                            this.grid[r][c] = updated.toString();
                            this.updateCell(r, c, updated.toString());
                        }
                    }
                }
            }
        } catch (error) {
            this.handleError(`Error removing number from cell (${row}, ${col}):`, error);
        }
    }

    updateNumbersAfterRemoval(row, col) {
        if (/\d/.test(this.grid[row][col])) {
            this.removeNumberFromCell(row, col);
        }
    }

    // ----------------------------------------------------------------
    //                          Drag Logic
    // ----------------------------------------------------------------

    startDrag(event) {
        if (!this.isDragMode) return;
        this.isDragging = true;

        const cell = event.target;
        const row = parseInt(cell.dataset.row, 10);
        const col = parseInt(cell.dataset.col, 10);

        this.toggleToBlack = this.grid[row][col] !== "#";
        this.toggleCell(row, col);
    }

    onDrag(event) {
        if (!this.isDragging || !this.isDragMode) return;
        const cell = document.elementFromPoint(event.clientX, event.clientY);
        if (cell && cell.dataset && cell.dataset.row && cell.dataset.col) {
            const row = parseInt(cell.dataset.row, 10);
            const col = parseInt(cell.dataset.col, 10);
            this.toggleCell(row, col);
        }
    }

    stopDrag() {
        this.isDragging = false;
    }

    toggleCell(row, col) {
        const cell = this.cells[`${row},${col}`];
        if (!cell) return;

        if (this.toggleToBlack && this.grid[row][col] !== "#") {
            this.updateCell(row, col, "", '#000', '#000');
            this.grid[row][col] = "#";
            this.updateNumbersAfterRemoval(row, col);
        } else if (!this.toggleToBlack && this.grid[row][col] === "#") {
            this.updateCell(row, col, "", '#444', '#fff');
            this.grid[row][col] = " ";
        }
    }

    // ----------------------------------------------------------------
    //                       Solving Methods
    // ----------------------------------------------------------------

    solveCrossword() {
        // Clear status
        const statusDisplay = document.getElementById('status-display');
        if (statusDisplay) {
            statusDisplay.value = '';
        }

        if (this.isSolving) {
            alert("A puzzle is already being solved. Please wait.");
            return;
        }
        this.isSolving = true;

        const solveButton = document.getElementById('solve-crossword-button');
        if (solveButton) solveButton.disabled = true;

        this.updateStatus("Solving...", true);

        this.solveCrosswordThread()
            .finally(() => {
                if (solveButton) solveButton.disabled = false;
                this.isSolving = false;
            });
    }

    async solveCrosswordThread() {
        const startTime = performance.now();
        try {
            if (!this.validateGrid()) return;

            this.generateSlots();
            if (Object.keys(this.slots).length === 0) {
                alert("No numbered slots found to solve.");
                this.updateStatus("Error: No numbered slots found to solve.");
                return;
            }

            this.randomizeDomains();
            this.updateStatus("Running AC-3 algorithm...");

            const ac3Success = this.ac3();
            const hasEmptyDomain = Object.values(this.domains).some(d => d.length === 0);

            if (!ac3Success || hasEmptyDomain) {
                this.updateStatus("AC-3 failed or domains wiped out. Attempting backtracking...");
            } else {
                this.updateStatus("Starting backtracking search...");
            }

            this.displayDomainSizes();

            this.recursiveCalls = 0;
            const backStart = performance.now();
            const result = this.backtrackingSolve();
            const backEnd = performance.now();

            const totalTime = (performance.now() - startTime) / 1000;

            if (result) {
                this.performanceData['Backtracking'] = {
                    time: (backEnd - backStart) / 1000,
                    calls: this.recursiveCalls
                };
                this.updateStatus("Solution found with backtracking.");
                this.displaySolution();
                this.displayWordList();
                this.updateStatus(`Total solving time: ${totalTime.toFixed(2)} seconds`);
                this.logPerformanceMetrics();
            } else {
                this.updateStatus("No possible solution found.");
                alert("No possible solution found for the current puzzle.");
            }
        } catch (error) {
            this.handleError("Error during solveCrosswordThread:", error);
        }
    }

    validateGrid() {
        if (this.grid.length === 0) {
            alert("The grid is empty. Please generate or load a grid.");
            this.updateStatus("Error: The grid is empty.");
            return false;
        }
        const cols = this.grid[0].length;
        for (let r = 1; r < this.grid.length; r++) {
            if (this.grid[r].length !== cols) {
                alert("The grid is not rectangular.");
                this.updateStatus("Error: The grid is not rectangular.");
                return false;
            }
        }
        return true;
    }

    generateSlots() {
        this.slots = {};
        this.domains = {};
        this.cellContents = {};

        const rows = this.grid.length;
        const cols = this.grid[0].length;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = this.grid[r][c];
                const key = `${r},${c}`;

                if (/[A-Z]/.test(val)) {
                    this.cellContents[key] = val;
                } else if (val !== "#" && val.trim() !== "") {
                    this.cellContents[key] = null;
                }
            }
        }

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (/\d/.test(this.grid[r][c])) {
                    if (c === 0 || this.grid[r][c - 1] === "#") {
                        const positions = this.getSlotPositions(r, c, "across");
                        if (positions.length >= 2) {
                            const slotName = `${this.grid[r][c]}ACROSS`;
                            this.slots[slotName] = positions;
                        }
                    }
                    if (r === 0 || this.grid[r - 1][c] === "#") {
                        const positions = this.getSlotPositions(r, c, "down");
                        if (positions.length >= 2) {
                            const slotName = `${this.grid[r][c]}DOWN`;
                            this.slots[slotName] = positions;
                        }
                    }
                }
            }
        }

        this.generateConstraints();
        this.setupDomains();
    }

    getSlotPositions(r, c, direction) {
        const positions = [];
        while (
            r < this.grid.length &&
            c < this.grid[0].length &&
            this.grid[r][c] !== "#"
        ) {
            positions.push([r, c]);
            if (direction === "across") {
                c++;
            } else {
                r++;
            }
        }
        return positions;
    }

    generateConstraints() {
        this.constraints = {};
        const positionMap = {};

        for (const slot in this.slots) {
            const positions = this.slots[slot];
            positions.forEach((pos, idx) => {
                const key = `${pos[0]},${pos[1]}`;
                if (!positionMap[key]) {
                    positionMap[key] = [];
                }
                positionMap[key].push({ slot, idx });
            });
        }

        for (const key in positionMap) {
            const overlaps = positionMap[key];
            if (overlaps.length > 1) {
                for (let i = 0; i < overlaps.length; i++) {
                    for (let j = i + 1; j < overlaps.length; j++) {
                        const slotA = overlaps[i].slot;
                        const idxA = overlaps[i].idx;
                        const slotB = overlaps[j].slot;
                        const idxB = overlaps[j].idx;

                        if (!this.constraints[slotA]) this.constraints[slotA] = {};
                        if (!this.constraints[slotA][slotB]) this.constraints[slotA][slotB] = [];
                        this.constraints[slotA][slotB].push([idxA, idxB]);

                        if (!this.constraints[slotB]) this.constraints[slotB] = {};
                        if (!this.constraints[slotB][slotA]) this.constraints[slotB][slotA] = [];
                        this.constraints[slotB][slotA].push([idxB, idxA]);
                    }
                }
            }
        }
    }

    setupDomains() {
        for (const slot in this.slots) {
            const positions = this.slots[slot];
            const length = positions.length;

            const pattern = positions.map(([r, c]) => {
                const key = `${r},${c}`;
                return this.cellContents[key] || '.';
            }).join('');

            const regex = new RegExp(`^${pattern}$`);
            const possibleWords = this.wordLengthCache[length] || [];
            const filtered = possibleWords.filter(word => regex.test(word));

            this.domains[slot] = filtered;
        }
    }

    ac3() {
        const queue = new Set();
        for (const slotA in this.constraints) {
            for (const slotB in this.constraints[slotA]) {
                queue.add(`${slotA},${slotB}`);
            }
        }

        while (queue.size > 0) {
            const [var1, var2] = Array.from(queue).shift().split(',');
            queue.delete(`${var1},${var2}`);

            if (this.revise(var1, var2)) {
                if (this.domains[var1].length === 0) {
                    return false;
                }
                for (const neighbor in this.constraints[var1]) {
                    if (neighbor !== var2) {
                        queue.add(`${neighbor},${var1}`);
                    }
                }
            }
        }
        return true;
    }

    revise(var1, var2) {
        let revised = false;
        const overlaps = this.constraints[var1][var2];
        const newDomain = this.domains[var1].filter(word1 => {
            return overlaps.some(([idx1, idx2]) => {
                return this.domains[var2].some(word2 => word1[idx1] === word2[idx2]);
            });
        });

        if (newDomain.length < this.domains[var1].length) {
            this.domains[var1] = newDomain;
            revised = true;
        }
        return revised;
    }

    backtrackingSolve(assignment = {}, cache = {}) {
        if (Object.keys(assignment).length === Object.keys(this.slots).length) {
            this.solution = { ...assignment };
            return true;
        }

        this.recursiveCalls++;
        const assignmentKey = JSON.stringify(Object.entries(assignment).sort());
        if (cache[assignmentKey] !== undefined) {
            return cache[assignmentKey];
        }

        const varToAssign = this.selectUnassignedVariable(assignment);
        if (!varToAssign) {
            cache[assignmentKey] = false;
            return false;
        }

        const orderedValues = this.orderDomainValues(varToAssign);

        for (const value of orderedValues) {
            if (this.isConsistent(varToAssign, value, assignment)) {
                assignment[varToAssign] = value;
                const inferences = this.forwardCheck(varToAssign, value, assignment);
                if (inferences !== false) {
                    if (this.backtrackingSolve(assignment, cache)) {
                        cache[assignmentKey] = true;
                        return true;
                    }
                }
                delete assignment[varToAssign];
                this.restoreDomains(inferences);
            }
        }

        cache[assignmentKey] = false;
        return false;
    }

    selectUnassignedVariable(assignment) {
        const unassigned = Object.keys(this.domains).filter(s => !(s in assignment));
        if (unassigned.length === 0) return null;

        let minSize = Infinity;
        let candidates = [];
        for (const slot of unassigned) {
            const size = this.domains[slot].length;
            if (size < minSize) {
                minSize = size;
                candidates = [slot];
            } else if (size === minSize) {
                candidates.push(slot);
            }
        }

        let maxDegree = -1;
        let finalCandidates = [];
        for (const slot of candidates) {
            const deg = this.constraints[slot] ? Object.keys(this.constraints[slot]).length : 0;
            if (deg > maxDegree) {
                maxDegree = deg;
                finalCandidates = [slot];
            } else if (deg === maxDegree) {
                finalCandidates.push(slot);
            }
        }

        return finalCandidates[Math.floor(Math.random() * finalCandidates.length)];
    }

    orderDomainValues(slot) {
        const domain = [...this.domains[slot]];
        const getFrequencyScore = (word) =>
            word.split('').reduce((acc, ch) => acc + (this.letterFrequencies[ch] || 0), 0);

        domain.sort((a, b) => getFrequencyScore(a) - getFrequencyScore(b));

        for (let i = domain.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [domain[i], domain[j]] = [domain[j], domain[i]];
        }
        return domain;
    }

    isConsistent(slot, word, assignment) {
        if (!this.wordMatchesPreFilledLetters(slot, word)) {
            return false;
        }
        const neighbors = this.constraints[slot] || {};
        for (const neighbor in neighbors) {
            if (neighbor in assignment) {
                if (!this.wordsMatch(slot, word, neighbor, assignment[neighbor])) {
                    return false;
                }
            } else {
                const viable = this.domains[neighbor].filter(
                    w => this.wordsMatch(slot, word, neighbor, w)
                );
                if (viable.length === 0) {
                    return false;
                }
            }
        }
        return true;
    }

    forwardCheck(slot, value, assignment) {
        const inferences = {};
        const neighbors = this.constraints[slot] || {};

        for (const neighbor in neighbors) {
            if (!(neighbor in assignment)) {
                const newDomain = this.domains[neighbor].filter(
                    w => this.wordsMatch(slot, value, neighbor, w)
                );
                if (newDomain.length === 0) {
                    return false;
                }
                inferences[neighbor] = this.domains[neighbor];
                this.domains[neighbor] = newDomain;
            }
        }
        return inferences;
    }

    restoreDomains(inferences) {
        if (!inferences) return;
        for (const v in inferences) {
            this.domains[v] = inferences[v];
        }
    }

    wordMatchesPreFilledLetters(slot, word) {
        const positions = this.slots[slot];
        for (let i = 0; i < positions.length; i++) {
            const [r, c] = positions[i];
            const key = `${r},${c}`;
            const preFilled = this.cellContents[key];
            if (preFilled && preFilled !== word[i]) {
                return false;
            }
        }
        return true;
    }

    wordsMatch(var1, word1, var2, word2) {
        const overlaps = this.constraints[var1][var2];
        for (const [idx1, idx2] of overlaps) {
            if (word1[idx1] !== word2[idx2]) {
                return false;
            }
        }
        return true;
    }

    randomizeDomains() {
        for (const slot in this.domains) {
            this.domains[slot] = this.shuffleArray(this.domains[slot]);
        }
        this.debugLog("Domains randomized.");
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    // ----------------------------------------------------------------
    //                      Display & Status
    // ----------------------------------------------------------------

    updateCell(row, col, value = null, fg = null, bg = null) {
        const cell = this.cells[`${row},${col}`];
        if (cell) {
            if (value !== null) cell.textContent = value;
            if (fg !== null) cell.style.color = fg;
            if (bg !== null) cell.style.backgroundColor = bg;
        }
    }

    updateStatus(message, isError = false) {
        const statusDisplay = document.getElementById('status-display');
        if (!statusDisplay) return;

        if (isError) {
            statusDisplay.style.color = 'red';
        } else {
            statusDisplay.style.color = '#222';
        }
        statusDisplay.value += `${message}\n`;
        statusDisplay.scrollTop = statusDisplay.scrollHeight;

        this.debugLog(message);
    }

    debugLog(message, ...args) {
        if (this.DEBUG) {
            console.log(`[DEBUG] ${message}`, ...args);
        }
    }

    handleError(userMessage, error) {
        console.error(userMessage, error);
        this.updateStatus(`${userMessage} ${error.message}`, true);
        alert(`${userMessage} See console for details.`);
    }

    // ----------------------------------------------------------------
    //                    Displaying the Solution
    // ----------------------------------------------------------------

    displaySolution() {
        try {
            for (const slot in this.solution) {
                const word = this.solution[slot];
                const positions = this.slots[slot];
                for (let i = 0; i < positions.length; i++) {
                    const [r, c] = positions[i];
                    this.updateCell(r, c, word[i], '#155724', '#d1e7dd');
                }
            }
            this.updateStatus("Solution displayed on the grid.");
            this.debugLog("Solution displayed on the grid.");
        } catch (error) {
            this.handleError("Error displaying solution:", error);
        }
    }

    displayWordList() {
        try {
            const acrossWords = [];
            const downWords = [];
            const sortedSlots = Object.keys(this.slots).sort((a, b) => {
                const numA = parseInt(a.match(/\d+/)[0], 10);
                const numB = parseInt(b.match(/\d+/)[0], 10);
                return numA - numB;
            });

            for (const slot of sortedSlots) {
                const word = this.solution[slot];
                if (word) {
                    const slotNum = slot.match(/\d+/)[0];
                    const entryText = `${slotNum}: ${word}`;
                    if (slot.endsWith("ACROSS")) {
                        acrossWords.push(entryText);
                    } else if (slot.endsWith("DOWN")) {
                        downWords.push(entryText);
                    }
                }
            }

            const acrossDisplay = document.getElementById('across-display');
            const downDisplay = document.getElementById('down-display');

            if (!acrossDisplay || !downDisplay) {
                throw new Error("Word list display elements not found. Make sure you replaced <textarea> with a container!");
            }

            acrossDisplay.innerHTML = '';
            downDisplay.innerHTML = '';

            acrossWords.forEach(entry => {
                const div = document.createElement('div');
                div.style.cursor = 'pointer';
                div.style.marginBottom = '5px';
                div.textContent = entry;
                div.addEventListener('click', () => {
                    const parts = entry.split(':');
                    if (parts.length === 2) {
                        const word = parts[1].trim();
                        // Now we do fallback
                        this.showDefinitionPopup(word);
                    }
                });
                acrossDisplay.appendChild(div);
            });

            downWords.forEach(entry => {
                const div = document.createElement('div');
                div.style.cursor = 'pointer';
                div.style.marginBottom = '5px';
                div.textContent = entry;
                div.addEventListener('click', () => {
                    const parts = entry.split(':');
                    if (parts.length === 2) {
                        const word = parts[1].trim();
                        this.showDefinitionPopup(word);
                    }
                });
                downDisplay.appendChild(div);
            });

        } catch (error) {
            this.handleError("Error displaying word lists:", error);
        }
    }

    logPerformanceMetrics() {
        try {
            for (const method in this.performanceData) {
                const data = this.performanceData[method];
                const msg = `${method} - Time: ${data.time.toFixed(4)}s, Recursive Calls: ${data.calls}`;
                this.updateStatus(msg);
                this.debugLog(msg);
            }
        } catch (error) {
            this.handleError("Error logging performance metrics:", error);
        }
    }

    displayDomainSizes() {
        try {
            this.updateStatus("Domain Sizes After Setup:");
            const sortedSlots = Object.keys(this.domains).sort((a, b) => {
                const numA = parseInt(a.match(/\d+/)[0], 10);
                const numB = parseInt(b.match(/\d+/)[0], 10);
                return numA - numB;
            });
            for (const slot of sortedSlots) {
                const size = this.domains[slot].length;
                this.updateStatus(`Domain for ${slot} has ${size} option(s).`);
            }
        } catch (error) {
            this.handleError("Error displaying domain sizes:", error);
        }
    }

    // ----------------------------------------------------------------
    //                     Word Lookup Handlers
    // ----------------------------------------------------------------

    handleSearchInput(event) {
        const query = event.target.value.trim().toUpperCase();
        const dropdown = document.getElementById('search-dropdown');
        const matchesCount = document.getElementById('matches-count');
        if (!dropdown || !matchesCount) return;

        if (query === "") {
            dropdown.style.display = 'none';
            matchesCount.textContent = "";
            return;
        }

        const sourceWords = this.words.map(w => w.toUpperCase());
        const allDictionaryWords = this.dictionary ? Object.keys(this.dictionary).map(k => k.toUpperCase()) : [];
        const combinedSet = new Set([...sourceWords, ...allDictionaryWords]);
        const combinedWords = Array.from(combinedSet);

        const matches = combinedWords.filter(word => word.startsWith(query)).sort();

        matchesCount.textContent = matches.length > 0
            ? `Found ${matches.length} match(es).`
            : "No matches found.";

        if (matches.length === 0) {
            dropdown.style.display = 'none';
            return;
        }

        const topMatches = matches.slice(0, 10);

        this.displaySearchResults(topMatches);
        dropdown.style.display = 'block';
    }

    displaySearchResults(matches) {
        const dropdown = document.getElementById('search-dropdown');
        const matchesCount = document.getElementById('matches-count');
        if (!dropdown) return;

        dropdown.innerHTML = '';

        matches.forEach(word => {
            const item = document.createElement('div');
            item.textContent = word;
            item.style.padding = '8px';
            item.style.cursor = 'pointer';
            item.style.borderBottom = '1px solid #eee';

            item.addEventListener('mouseover', () => item.style.backgroundColor = '#f1f1f1');
            item.addEventListener('mouseout', () => item.style.backgroundColor = '#fff');
            item.addEventListener('click', () => {
                const searchInput = document.getElementById('word-search-input');
                if (searchInput) {
                    searchInput.value = word;
                }
                dropdown.style.display = 'none';
                if (matchesCount) {
                    matchesCount.textContent = "Found 1 match.";
                }
                // Show the definition (with fallback) for that word
                this.showDefinitionPopup(word);
            });

            dropdown.appendChild(item);
        });
    }
}
