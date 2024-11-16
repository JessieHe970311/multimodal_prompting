<template>
  <div>
    <div class="HeaderPanel">
      <div class="title">POEM</div>
        <div style="width: 30px"></div>
        <div class="sentiment-block positive"></div>
        <div class="sentiment-label">POSITIVE</div>
        <div class="sentiment-block neutral"></div>
        <div class="sentiment-label">NEUTRAL</div>
        <div class="sentiment-block negative"></div>
        <div class="sentiment-label">NEGATIVE</div>

        <div style="width: 50px"></div>
        <div class="sentiment-block false"></div>
        <div class="sentiment-label2">FALSE</div>
        <div class="sentiment-block true"></div>
        <div class="sentiment-label2">TRUE</div>
    </div>
    <el-row :gutter="0">
      <el-col :span="6">
        <div class="LeftPanel myPanel">
          <div class="Panelname">Prompt Panel</div>
          <PromptPanel :uploadedPrinciple="principleMessage" :uploadedKshot="kshotupload" :initial="results"
            @currentp="recievePrompt" />
          <PrincipleRecommendation :selectKshotData="selectKshot" @principledata="receivePrinciple" />
          <KshotExample :kshotExampleData="selectKshotExample" :example="kshot" @kshotdata="recieveKshot" />
        </div>
      </el-col>
      <el-col :span="12">
        <div class="MiddlePanel myPanel">
          <div class="Panelname">Reasoning Panel</div>
          <ReasoningPath :allInstancedata="results" @instancedata="receiveInstance" />
          <ReasoningPanel :allInstancedata="results" :kshotInstance="kshot" :testInstance="instanceMessage"
            @selecttestdata="testData" @selectkshotdata="kshotData" @selectkshotexampledata="kshotexampleData"
            @retrievetestdata="retrievedata" />
        </div>
      </el-col>
      <el-col :span="6">
        <div class="RightPanel myPanel">
          <div class="Panelname">Evaluation Panel</div>
          <PromptHistory :lastPrompt="submitPrompt" :kshotExampleData="kshot" :allInstancedata="results" @newresults="getnewResults" />
          <ModelPerformance :allInstancedata="results" :newdata="newperformance"/>
          <TestPanel :example="results" :selectTestData="selectTest" :retrieveInstance="retrieve" />
        </div>
      </el-col>
    </el-row>
    <!-- </el-container>
    <el-divider></el-divider> -->
  </div>
</template>

<script setup>
import { ElRow, ElCol } from 'element-plus'
</script>

<script>
import PromptPanel from './components/PromptPanel.vue'
import ReasoningPanel from './components/ReasoningPanel.vue'
import PrincipleRecommendation from './components/PrincipleRecommendation.vue'
import PromptHistory from './components/PromptHistory.vue'
import TestPanel from './components/TestPanel.vue'
import ModelPerformance from './components/ModelPerformance.vue'
import ReasoningPath from './components/ReasoningPath.vue'
import KshotExample from './components/KshotExample.vue'

import requesthelp from "./service/request.js";

export default {
  name: 'App',
  components: {
    PromptPanel,
    ReasoningPanel,
    PrincipleRecommendation,
    PromptHistory,
    TestPanel,
    ModelPerformance,
    ReasoningPath,
    KshotExample
  },
  data() {
    return {
      principleMessage: [],
      instanceMessage: [],
      selectTest: [],
      selectKshot: [],
      selectKshotExample: [],
      submitPrompt: [],
      results: {},
      kshot: {},
      kshotupload: {},
      retrieve: {},
      newperformance: {}
    }
  },
  mounted() {
    this.getinitial()
    this.getkshot()
  },
  methods: {
    async getinitial() {
      var results = await requesthelp.axiosGet("/load_initial_prompt_and_result")
      this.results = results;
      console.log("initial", results)
    },
    async getkshot() {
      var results = await requesthelp.axiosGet("/load_k_shot_example")
      this.kshot = results;
      console.log("kshot", this.kshot)
    },
    receivePrinciple(message) {
      this.principleMessage = message;
    },
    receiveInstance(message) {
      this.instanceMessage = message;
    },
    testData(message) {
      this.selectTest = message;
    },
    kshotData(message) {
      this.selectKshot = message;
    },
    kshotexampleData(message) {
      this.selectKshotExample = message;
    },
    recievePrompt(message) {
      this.submitPrompt = message;
    },
    recieveKshot(message) {
      this.kshotupload = message
    },
    retrievedata(message) {
      this.retrieve = message
    },
    getnewResults(message){
      this.newperformance = message
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;  
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 10px;
}

.LeftPanel {
  border-right: 0 !important;
}

.MiddlePanel {
  border-right: 0 !important;
}

/* .RightPanel {
  border-right: 0;
} */

.myPanel {
  border: 2px solid rgb(216, 216, 216);
  height: 95vh;
}

.viewbottom {
  border-bottom: 2px solid rgb(216, 216, 216);
}

.viewbottom {
  padding-top: 2px;
}

.Panelname {
  text-align: left;
  font-weight: bold;
  padding-left: 10px;
  padding-top: 4px;
  padding-bottom: 0px;
  font-size: 20px;
  height: 2vh;
  text-transform: uppercase;
  /* font-family: cursive; */
}

.selectTitle {
  text-align: left;
  padding-left: 10px;
  padding-top: 5px;
  padding-bottom: 5px;
  font-weight: 500;
}

.header-row {
  margin-bottom: 10px;
  margin-top: 10px;
}

.HeaderPanel {
  text-align: left;
  display: block;
  font-size: 1.8em;
  margin-block-start: 0em;
  margin-block-end: 0em;
  margin-inline-start: 0px;
  margin-inline-end: 0px;
  font-weight: bold;
  background-color: #333954;
  padding-top: 3px;
  padding-bottom: 3px;
  padding-left: 10px;
  display: flex;
  align-items: center;
  /* 确保标题和形状垂直居中 */
  height: 3vh;
  overflow: hidden;
}

.title {
  color: white;
  font-family: cursive;
  font-weight: 900;
  margin-right: 20px;
  /* 在标题和形状之间添加一些间隔 */
  font-size: 24px;
  /* 标题大小，根据需要调整 */
  /* 其他你需要的样式 */
}

.positive {
  background: #CB827C !important;
}

.neutral {
  background: #F4EDE1 !important;
}

.negative {
  background: #8A9DBC !important;
}

.true {
  background: #DADCD0 !important;
}

.false {
  background: #CCBEB8 !important;
}

.sentiment-labels {
  display: flex;
  flex-direction: column;
  /* Stack the labels vertically */
}

.sentiment-label {
  font-size: 14px;
  /* margin-bottom: 2px; */
  color: white;
  padding: 5px;
  width: 80px;
  /* Space below each label */
}

.sentiment-block {
  width: 20px;
  height: 20px;
  /* margin-bottom: 5px; */
  /* Space between blocks */
  border-radius: 5px;
  /* Rounded corners */
}

.sentiment-label2 {
  font-size: 14px;
  color: white;
  padding: 5px;
  width: 70px;
}
</style>