// WordListProvider.js
export class WordListProvider {
  constructor({ basePath = "Data/words_by_length", uppercase = true } = {}) {
    this.basePath = basePath;
    this.uppercase = uppercase;

    // length -> Promise<string[]>
    this._lengthPromises = new Map();

    // length -> string[]
    this._lengthWords = new Map();
  }

  async getWordsOfLength(len) {
    if (this._lengthWords.has(len)) return this._lengthWords.get(len);

    if (!this._lengthPromises.has(len)) {
      const p = (async () => {
        const url = `${this.basePath}/words-${len}.txt`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
        const text = await resp.text();

        // Split lines, filter empties
        let words = text.split(/\r?\n/).map(w => w.trim()).filter(Boolean);
        if (this.uppercase) words = words.map(w => w.toUpperCase());

        this._lengthWords.set(len, words);
        return words;
      })().catch(err => {
        // If it failed, allow retry later
        this._lengthPromises.delete(len);
        throw err;
      });

      this._lengthPromises.set(len, p);
    }

    return await this._lengthPromises.get(len);
  }

  // Optional: preload a set of lengths in parallel
  async preloadLengths(lengths) {
    await Promise.all([...new Set(lengths)].map(L => this.getWordsOfLength(L)));
  }
}
