import { createApp } from 'vue'
import App from './App.vue'
import 'element-plus/dist/index.css'


import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";

const app = createApp(App).component("font-awesome-icon", FontAwesomeIcon);

app.mount('#app');
