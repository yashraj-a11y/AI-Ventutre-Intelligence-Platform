// Day 19 · Phase C/J — app shell + routing. Portals are gated behind ProtectedRoute (auth-ready).
import { Route, Routes } from "react-router-dom";
import Footer from "./components/Layout/Footer";
import Header from "./components/Layout/Header";
import ProtectedRoute from "./components/ProtectedRoute";
import FounderPortal from "./pages/FounderPortal";
import Home from "./pages/Home";
import InvestorPortal from "./pages/InvestorPortal";
import Login from "./pages/Login";

export default function App() {
  return (
    <div className="flex min-h-full flex-col">
      <Header />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/founder" element={<ProtectedRoute><FounderPortal /></ProtectedRoute>} />
          <Route path="/investor" element={<ProtectedRoute><InvestorPortal /></ProtectedRoute>} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
