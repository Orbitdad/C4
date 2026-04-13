```typescript
import { createBrowserHistory } from 'history';
import { createRouter, createWebHistory } from 'vue-router';
import { createApp } from 'vue';

// Define the main application component
const App = createApp({
  template: `
    <div class="container">
      <h1>Portfolio Website</h1>
      <router-view />
    </div>
  `,
});

// Create a router instance with history mode
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('./views/Home.vue'),
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('./views/About.vue'),
    },
  ],
});

// Create a browser history instance
const browserHistory = createBrowserHistory();

// Define the main application entry point
App.config.globalProperties.$router = router;
App.config.globalProperties.$browserHistory = browserHistory;

// Create a Vite plugin to enable hot reloading and production mode
import { defineConfig } from 'vite';
export default defineConfig({
  plugins: [
    // Enable hot reloading for development mode
    {
      name: 'hot-reload',
      build: (config) => {
        config.hot = true;
        return config;
      },
    },
    // Enable production mode for production builds
    {
      name: 'production',
      build: (config) => {
        config.mode = 'production';
        return config;
      },
    },
  ],
});

// Define the global styles
import './styles/global.css';

// Create a Vite plugin to enable CSS modules and hot reloading
import { defineConfig } from 'vite';
export default defineConfig({
  plugins: [
    // Enable CSS modules for development mode
    {
      name: 'css-modules',
      build: (config) => {
        config.module.rules.push({
          test: /\.css$/,
          use: ['css-loader'],
        });
        return config;
      },
    },
    // Enable hot reloading for development mode
    {
      name: 'hot-reload',
      build: (config) => {
        config.hot = true;
        return config;
      },
    },
  ],
});

// Define the main application component
const App = createApp({
  template: `
    <div class="container">
      <h1>Portfolio Website</h1>
      <router-view />
    </div>
  `,
});

// Create a router instance with history mode
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('./views/Home.vue'),
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('./views/About.vue'),
    },
  ],
});

// Create a browser history instance
const browserHistory = createBrowserHistory();

// Define the main application entry point
App.config.globalProperties.$router = router;
App.config.globalProperties.$browserHistory = browserHistory;

// Export the App component as the main application entry point
export default App;
```