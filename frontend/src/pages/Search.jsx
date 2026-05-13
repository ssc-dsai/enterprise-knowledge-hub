import { useState } from "react";
import Result from "../components/Result";

export default function Search() {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    setLoading(true);
    setResults([]);

    const url = `http://127.0.0.1:8000/database/search?query=${encodeURIComponent(
        query
    )}&limit=${limit}`;

    console.log("Calling:", url);

    try {
        const res = await fetch(url);
        const data = await res.json();

        console.log("RAW RESPONSE:", data);

        let items = [];

        if (Array.isArray(data)) {
            items = data;
        } else if (data.results) {
            items = data.results;
        } else if (data.items) {
            items = data.items;
        }

        console.log("PARSED ITEMS:", items);

        setResults(items);
    } catch (err) {
        console.error(err);
    }

    setLoading(false); 
};

  return (
    <div>
      <h2 style={{ marginBottom: 10 }}>Search</h2>

      <form
        onSubmit={(e) => {
            e.preventDefault();
            search();
        }}
      >
        <div
            style={{
            display: "flex",
            gap: 10,
            marginBottom: 20,
            alignItems: "center",
            }}
        >
            <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Please enter your search query..."
            required
            style={{
                flex: 1,
                padding: "10px",
                borderRadius: 6,
                border: "1px solid #ccc",
            }}
            /> 
            <button
            type="submit"
            style={{
                padding: "10px 16px",
                borderRadius: 6,
                border: "none",
                background: "#333",
                color: "white",
                cursor: "pointer",
            }}
            >
            Search
            </button>

            <label>
                Results: <strong>{limit}</strong>
            </label>

            <input
            type="range"
            min="1"
            max="20"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
            /> 
        </div>
      </form>

      {loading ? (
        <p>Searching...</p>
      ) : (results.map((item, i) => (
        <Result key={i} item={item} index={i} />
      ))
    )}      
    </div>
  );
}
