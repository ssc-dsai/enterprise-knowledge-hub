import { useState } from "react";

export default function Result({ item, index}) {

  let file = "";
  let content = "";
  let node = "";
  let score = 0;

  if (Array.isArray(item)) {
    [file, content, node, score] = item;
  } else if (typeof item === "object") {
    file = item.title || item.name || item.path || "";
    content = item.content || item.snippet || item.excerpt || "";
    node = item.node ?? item.chunk_index ?? "";
    score = item.score ?? item.similarity ?? 0;
  }

  const load = async () => {
    setExpanded(true);

    const fileName = (file || "").split("/").pop();

    const res = await fetch(
      `http://127.0.0.1:8000/database/retrieve/${encodeURIComponent(
        fileName
      )}?source=wiki`
    );

    const text = await res.text();
    setFull(text);
  };

  return (
    <div
      style={{
        border: "1px solid #e5e5e5",
        borderRadius: 8,
        padding: 15,
        marginBottom: 12,
        background: "#fff",
      }}
    >
      <div
        style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 8,
        }}
    >
      <div>
        <div style={{ fontSize: 16 }}>
            <strong>Result {index + 1}:</strong>
        </div>
        <button
          onClick={load}
          style={{
            background: "none",
            border: "none",
            padding: 0,
            fontSize: 15,
            fontWeight: "bold",
            cursor: "pointer",
            color: "#1490C9",
          }}
        >
         {file} (node {node})
        </button>
      </div>

      <div style={{ fontSize: 15, color: "#6e6b6b" }}>
        <strong>Score: {score?.toFixed?.(2)}</strong>
      </div>
    </div>

    <div style={{ fontSize: 16}}>
      {content}
    </div>
   </div>
  );
}
