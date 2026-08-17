// Site footer: project links and the honest scope line.
import { Link } from "react-router-dom";
import { LINKS } from "../../lib/examples";
import { Logo } from "../ui";

export default function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 md:flex-row md:items-start md:justify-between">
        <div className="max-w-sm">
          <Link to="/" className="flex items-center gap-2.5">
            <Logo className="h-7 w-7" />
            <span className="font-bold text-slate-900">AI Venture Intelligence</span>
          </Link>
          <p className="mt-3 text-sm text-slate-500">
            A portfolio project: decision support trained on 2009–2017 Kickstarter campaigns. Predictions are
            model estimates, not financial advice.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-8 text-sm">
          <div>
            <p className="font-semibold text-slate-900">Product</p>
            <ul className="mt-3 space-y-2 text-slate-500">
              <li><Link to="/founder" className="hover:text-indigo-600">Founder portal</Link></li>
              <li><Link to="/investor" className="hover:text-indigo-600">Investor portal</Link></li>
              <li><a href="/#how-it-works" className="hover:text-indigo-600">How it works</a></li>
            </ul>
          </div>
          <div>
            <p className="font-semibold text-slate-900">Connect</p>
            <ul className="mt-3 space-y-2 text-slate-500">
              <li><a href={LINKS.linkedin} target="_blank" rel="noreferrer" className="hover:text-indigo-600">LinkedIn</a></li>
            </ul>
          </div>
        </div>
      </div>
      <div className="border-t border-slate-100 py-4 text-center text-xs text-slate-400">
        © 2026 yashraj-a11y · MIT License
      </div>
    </footer>
  );
}
