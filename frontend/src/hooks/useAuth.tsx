// Day 19 · Phase J — auth-READY (not auth-built). Real context + role + token slot, but a STUB login:
// no credentials, no JWT, no backend call. To make it real, replace `login()` with a
// `POST /auth/login` that returns a JWT, store the token, and decode the role from it — everything
// else (ProtectedRoute, role state, the token slot, localStorage persistence) stays the same.
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export type Role = "founder" | "investor";

interface AuthState {
  role: Role | null;
  token: string | null;
}
interface AuthContextValue extends AuthState {
  isAuthenticated: boolean;
  login: (role: Role) => void;
  logout: () => void;
}

const STORAGE_KEY = "avip.auth";
const AuthContext = createContext<AuthContextValue | null>(null);

function loadState(): AuthState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as AuthState;
  } catch {
    /* ignore malformed storage */
  }
  return { role: null, token: null };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(loadState);

  // STUB: sets a role + placeholder token. Swap for a real JWT flow later.
  const login = useCallback((role: Role) => {
    const next: AuthState = { role, token: `stub-token-${role}` };
    setState(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }, []);

  const logout = useCallback(() => {
    setState({ role: null, token: null });
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ ...state, isAuthenticated: state.token != null, login, logout }),
    [state, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
