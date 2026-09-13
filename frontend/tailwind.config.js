/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Wafer map die states, used by both the canvas renderer and the legend
        // so a colour change cannot make them disagree.
        die: {
          outside: "#0f172a",
          pass: "#1e5f8c",
          fail: "#f0663f",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
