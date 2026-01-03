// main.js
import { CrosswordSolver } from './CrosswordSolver.js';
import { loadWordNetDictionary } from './WordNetLoader.js'; 
// ^ We'll create a WordNetLoader.js that merges all JSON files into one aggregator object.
//   Or if you'd prefer, we can do inline fetches of each file. For now, let's assume a single function does it all.

document.addEventListener('DOMContentLoaded', async () => {
  try {
    // 1. Load the wordnet dictionary aggregator
    const dictionary = await loadWordNetDictionary();

    // 2. Instantiate and initialize the solver
    const crosswordSolver = new CrosswordSolver();

    // 3. Pass the dictionary into the solver
    crosswordSolver.setDictionary(dictionary);

    // 4. Now run the usual init() method
    crosswordSolver.init();

  } catch (err) {
    console.error("Error loading dictionary or initializing solver:", err);
    alert("Fatal error. Check console for details.");
  }
});
