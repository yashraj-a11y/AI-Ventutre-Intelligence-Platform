// Day 19 · Phase C/J — app header: brand, role nav + auth state (sign in / role + logout), mobile menu.
import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { Logo } from "../ui";

const link = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-3 py-2 text-left text-sm font-medium transition ${
    isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
  }`;

export default function Header() {
  const { isAuthenticated, role, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);

  const nav = (
    <>
      <NavLink to="/founder" className={link} onClick={close}>Founder</NavLink>
      <NavLink to="/investor" className={link} onClick={close}>Investor</NavLink>
      {isAuthenticated ? (
        <>
          <span className="hidden rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium capitalize text-slate-500 md:inline">{role}</span>
          <button onClick={() => { logout(); close(); }} className={link({ isActive: false })}>Log out</button>
        </>
      ) : (
        <NavLink to="/login" className={link} onClick={close}>Sign in</NavLink>
      )}
    </>
  );

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link to="/" className="flex items-center gap-2.5" onClick={close}>
          <Logo className="h-8 w-8" />
          <span className="text-sm font-bold tracking-tight text-slate-900 sm:text-base">AI Venture Intelligence</span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">{nav}</nav>

        <button
          className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 md:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
        >
          <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            {open ? <path d="M6 6l12 12M18 6 6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </div>

      {open && <nav className="flex flex-col gap-1 border-t border-slate-100 px-4 py-3 md:hidden">{nav}</nav>}
    </header>
  );
}
