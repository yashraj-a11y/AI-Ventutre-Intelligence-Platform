// Day 19 · Phase J — gate a route behind authentication. With the stub login this only checks that a
// role was chosen; with real auth it would additionally verify the JWT (and could enforce a role).
import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
