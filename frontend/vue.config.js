const { defineConfig } = require('@vue/cli-service')
const webpack = require('webpack');

module.exports = defineConfig({
  transpileDependencies: true,
  configureWebpack: {
    plugins: [
      new webpack.DefinePlugin({
        // Define other feature flags if needed
        __VUE_OPTIONS_API__: true, // if you're using Vue's Options API
        __VUE_PROD_DEVTOOLS__: false, // if you want to disable devtools in production
        __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false // disable hydration mismatch details in production
      })
    ]
  },
  devServer: {
    proxy: {
      '/api': {
        target: 'https://c5176a5fb298.ngrok.app', // Your backend URL
        changeOrigin: true,
        pathRewrite: { '^/api': '' } // Optional: rewrite the path if needed
      }
    },
    allowedHosts: 'all',
    //https: true,
    client: {
      // Configure WebSocket settings to use the ngrok URL
      webSocketURL: 'wss://110299e21f62.ngrok.app/ws' // Replace with your ngrok URL
    }
  }
})

