import { useEffect, useState } from "react";

export default function Status() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/frontend/status")
      .then((res) => res.text())
      .then((html) => {
        document.getElementById("table").innerHTML = html;
      });
  }, []);

  return (
    <div>
      <div id="table"></div>
    </div>
  );
}