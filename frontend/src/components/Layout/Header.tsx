// Day 19 · Phase C/J — app header: role nav + auth state (sign in / role + logout).
import { NavLink } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";

const link = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-3 py-1.5 text-sm font-medium transition ${
    isActive ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-100"
  }`;

export default function Header() {
  const { isAuthenticated, role, logout } = useAuth();

  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3">
        <NavLink to="/" className="text-sm font-bold text-slate-900">AI Venture Intelligence</NavLink>
        <nav className="flex items-center gap-2">
          <NavLink to="/founder" className={link}>Founder</NavLink>
          <NavLink to="/investor" className={link}>Investor</NavLink>
          {isAuthenticated ? (
            <>
              <span className="ml-1 hidden rounded-full bg-slate-100 px-2 py-1 text-xs font-medium capitalize text-slate-500 sm:inline">{role}</span>
              <button onClick={logout} className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-100">Logout</button>
            </>
          ) : (
            <NavLink to="/login" className={link}>Sign in</NavLink>
          )}
        </nav>
      </div>
    </header>
  );
}
