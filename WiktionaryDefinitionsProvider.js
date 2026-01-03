// WiktionaryDefinitionsProvider.js
export class WiktionaryDefinitionsProvider {
  constructor({ basePath = "Data/defs_by_length" } = {}) {
    this.basePath = basePath;

    // len -> defs map object { "WORD": [ {pos, definitions}, ... ] }
    this._cache = new Map();
    // len -> Promise resolving to defs map
    this._promises = new Map();
  }

  async _loadLength(len) {
    if (this._cache.has(len)) return this._cache.get(len);

    if (!this._promises.has(len)) {
      const p = (async () => {
        const url = `${this.basePath}/defs-${len}.json`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
        const defsMap = await resp.json();
        this._cache.set(len, defsMap);
        return defsMap;
      })().catch(err => {
        // allow retry after failure
        this._promises.delete(len);
        throw err;
      });

      this._promises.set(len, p);
    }

    return await this._promises.get(len);
  }

  async lookup(rawWord) {
    if (!rawWord) return null;
    const word = rawWord.toUpperCase();
    const len = word.length;

    const defsMap = await this._loadLength(len);
    return defsMap[word] || null;
  }
}
