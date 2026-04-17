# Character Lab (web)

Vite + React + TypeScript front end for Character Lab.

## Prerequisites

- [Node.js](https://nodejs.org/) (LTS recommended) and npm

## Environment variables

The app reads the Gemini API key from Vite’s env surface as `VITE_GEMINI_API_KEY`.

1. In the `web` folder, create a file named `.env` (same directory as `package.json`).
2. Add your key in this form (no spaces around `=`):

   ```bash
   VITE_GEMINI_API_KEY=your_key_here
   ```

3. Obtain a key from [Google AI Studio](https://aistudio.google.com/apikey) (or your course’s documented source). Treat it like a password: do not commit `.env` or share the key publicly.

Vite only exposes variables prefixed with `VITE_` to client code. After changing `.env`, restart the dev server so the new value is picked up.

## Install and run

From the `web` directory:

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

### Other scripts

| Command        | Description                |
| -------------- | -------------------------- |
| `npm run build`   | Typecheck and production build |
| `npm run preview` | Serve the production build locally |
| `npm run lint`    | Run ESLint                 |
