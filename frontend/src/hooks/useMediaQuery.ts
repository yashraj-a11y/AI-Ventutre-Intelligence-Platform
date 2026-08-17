// True while a CSS media query matches (e.g. "(max-width: 639px)"); updates on resize/rotation.
import { useSyncExternalStore } from "react";

export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** Phone-width layout (below Tailwind's `sm` breakpoint). */
export const useIsPhone = () => useMediaQuery("(max-width: 639px)");
