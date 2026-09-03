// @ts-check
import { defineConfig } from 'astro/config';
import nodePolyfills from '@rolldown/plugin-node-polyfills';

// https://astro.build/config
export default defineConfig({
  vite: {
    plugins: [nodePolyfills()],
    resolve: {
      alias: {
        events: 'events/',
        url: 'url/',
      },
    },
  },
});
