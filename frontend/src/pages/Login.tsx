// Day 19 · Phase J — role select (STUB "login"). No credentials — this demonstrates the auth-ready
// structure. A real sign-in form + JWT replaces this without touching the routes or the context API.
import { useNavigate } from "react-router-dom";
import { useAuth, type Role } from "../hooks/useAuth";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const choose = (role: Role) => {
    login(role);
    navigate(`/${role}`, { replace: true });
  };

  return (
    <div className="mx-auto max-w-md px-4 py-20 text-center">
      <h1 className="text-2xl font-bold text-slate-900">Sign in</h1>
      <p className="mt-2 text-sm text-slate-500">Choose a role to continue.</p>
      <div className="mt-8 grid grid-cols-2 gap-4">
        <button onClick={() => choose("founder")} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-indigo-400">
          <div className="text-lg font-semibold text-slate-900">Founder</div>
          <div className="mt-1 text-xs text-slate-500">Predict, explain, improve</div>
        </button>
        <button onClick={() => choose("investor")} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-indigo-400">
          <div className="text-lg font-semibold text-slate-900">Investor</div>
          <div className="mt-1 text-xs text-slate-500">Verdict, comparables, due diligence</div>
        </button>
      </div>
      <p className="mt-6 text-xs text-slate-400">
        Demo: no password. This is the auth-ready structure — real JWT authentication drops in without
        changing the routes or components.
      </p>
    </div>
  );
}
