import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Search from "./pages/Search";
import Status from "./pages/Status";

export default function App() {
  return (
    <BrowserRouter>
      <div
        style={{
          maxWidth: 1200,
          margin: "40px auto",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <nav
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 20,
          }}
        >
          <h1>Enterprise Knowledge Hub Endpoint Test</h1>
          <div>
            <Link to="/">Search</Link> |{" "}
            <Link to="/status">Run History and Status</Link>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Search />} />
          <Route path="/status" element={<Status />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
