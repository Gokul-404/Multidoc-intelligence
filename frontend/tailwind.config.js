/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        sentinel: {
          50: "#f0f4ff",
          100: "#dbe4fe",
          200: "#bfd0fd",
          300: "#93b3fc",
          400: "#608efa",
          500: "#3b6cf6",
          600: "#254feb",
          700: "#1d3ec7",
          800: "#1e35a1",
          900: "#1e2f7f",
          950: "#0b122c",
        },
        slate: {
          850: "#161f30",
          950: "#0b0f19",
        },
        emerald: {
          450: "#10b981",
        },
      },
      fontFamily: {
        sans: ["Outfit", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        glow: "0 0 25px -5px rgba(59, 130, 246, 0.4)",
        glowEmerald: "0 0 25px -5px rgba(16, 185, 129, 0.4)",
        glowRose: "0 0 25px -5px rgba(244, 63, 94, 0.4)",
      },
    },
  },
  plugins: [],
}
