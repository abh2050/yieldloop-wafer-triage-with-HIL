module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended-type-checked",
    "plugin:react-hooks/recommended",
  ],
  parser: "@typescript-eslint/parser",
  parserOptions: { project: ["./tsconfig.json"], tsconfigRootDir: __dirname },
  plugins: ["@typescript-eslint"],
  ignorePatterns: ["dist", ".eslintrc.cjs", "*.config.ts", "*.config.js"],
  rules: {
    "@typescript-eslint/no-misused-promises": ["error", { checksVoidReturn: false }],
  },
};
