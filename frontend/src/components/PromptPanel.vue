<template>
    <div class="promptpanelContainer viewbottom view">
        <div class="info">
            <div class="selections">
                <div>Dataset:</div>
                <el-select v-model="selectedDataset" placeholder="Select">
                    <el-option v-for="item in datasetsList" :key="item.value" :label="item.label" :value="item.value">
                    </el-option>
                </el-select>
                <div>Model:</div>
                <el-select v-model="selectedModel" placeholder="Select">
                    <el-option v-for="item in modelsList" :key="item.value" :label="item.label" :value="item.value">
                    </el-option>
                </el-select>
            </div>
            <div class="selectTitle">Prompt Editor</div>
        </div>
        <div class="currentpromptdiv">
            <!-- <div class="selectTitle">Prompt Editor</div> -->
            <div class="switchbutton">
                <span v-if="isCollapseMode">All-in-one Mode</span>
                <span v-else><b>All-in-one Mode</b></span>
                <el-switch v-model="isCollapseMode" inactive-color="#c9c9c9" active-color="#13ce66"
                    @change="handleSwitchChange">
                </el-switch>
                <span v-if="isCollapseMode"><b>Structured Mode</b></span>
                <span v-else>Structured Mode</span>
                <div class="popuptemplate">
                    <el-popover placement="bottom" :width="400" trigger="click">
                        <template #reference>
                            <el-button type="info" plain style="margin-right: 16px">Template</el-button>
                        </template>
                        <div style="height:auto;">
                            <el-radio-group v-model="selected">
                                <el-radio v-for="(item, index) in templateData" :key="'radio' + index"
                                    :label="'radio' + index">
                                    <el-card>
                                        <div class="mywarp" v-for="details in item"
                                            :key="'detail' + index + details.key">
                                            <b>{{ details.value }}: </b>
                                            {{ details.text }}
                                        </div>
                                    </el-card>
                                </el-radio>
                            </el-radio-group>
                        </div>
                    </el-popover>
                </div>

                <div class="promptButtons">
                    <el-button type="primary" @click="submit">Submit</el-button>
                </div>
            </div>
            <div class="promptcontainer">
                <div class="currentPrompt" v-if="isCollapseMode">
                    <el-collapse v-model="activeNames">
                        <el-collapse-item v-for="item in currentprompt" :key="item.value" :name="item.name"
                            :title="item.name" class="collapse-prompt-custom">
                            <!-- <div>{{ item.prompt_text }}</div> -->
                            <div v-if="Array.isArray(item.text)">
                                <!-- If item.text is an array, loop through it and display each item -->
                                <!-- <div> -->
                                <div v-for="(text, index) in item.text" :key="index">
                                    <el-input class="promptinput" type="textarea"
                                        :placeholder="`Enter your prompt here... `"
                                        v-model="item.text[index]" autosize>
                                    </el-input>
                                </div>

                                <el-input class="promptinput" type="textarea" placeholder="Enter your prompt here..."
                                        v-model="myprinciple" autosize>
                                </el-input>
                                <!-- </div> -->
                            </div>

                            <div v-else-if="isObject(item.text)">
                                <!-- If item.text is an object, display each key-value pair -->
                                <div v-for="(value, key) in item.text" :key="key">
                                    <label>{{ key }}:</label>
                                    <el-input class="promptinput" type="textarea" :placeholder="`Enter ${key} here...`"
                                        v-model="item.text[key]" autosize>
                                    </el-input>
                                </div>
<!-- 
                                <el-input class="promptinput" type="textarea" placeholder="Enter your prompt here..."
                                        v-model="mykshot" autosize>
                                </el-input> -->
                            </div>

                            <el-input v-else class="promptinput" type="textarea" placeholder="Enter your prompt here..."
                                v-model="item.text" autosize>
                            </el-input>
                        </el-collapse-item>
                    </el-collapse>
                </div>
                <div class="currentPrompt" v-else>
                    Row prompt
                    <el-input class="rawinput" type="textarea" placeholder="Enter your prompt here..."
                        v-model="rawprompt">
                    </el-input>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import {
    ElSelect,
    ElOption,
    ElInput,
    ElButton,
    ElCollapse,
    ElCollapseItem,
    ElSwitch,
    ElPopover,
    ElCard,
    ElRadio,
    ElRadioGroup
} from 'element-plus'
</script>

<script>
import requesthelp from "../service/request.js";

export default {
    props: ['uploadedPrinciple', 'initial', 'uploadedKshot'],
    data() {
        return {
            prompt: '',
            selectedModel: '',
            modelsList: [{
                value: 'ChatGPT',
                label: 'ChatGPT'
            },
            {
                value: 'gpt-3.5-turbo-0125CMU-MOSEI',
                label: 'gpt-3.5-turbo-0125'
            },
            {
                value: 'LLaMA1-7B',
                label: 'LLaMA1-7B'
            },
            {
                value: 'LLaMA1-13B',
                label: 'LLaMA1-13B'
            },
            {
                value: 'LLaMA2-7B',
                label: 'LLaMA2-7B'
            }
            ],
            selectedDataset: '',
            datasetsList: [{
                value: 'CMU-MOSEI',
                label: 'CMU-MOSEI'
            },
            {
                value: 'ScienceOA',
                label: 'ScienceOA'
            },
            {
                value: 'MVSA-S',
                label: 'MVSA-S'
            },
            {
                value: 'MVSA-M',
                label: 'MVSA-M'
            }
            ],
            promptname: ['System Prompt', 'Task Prompt', 'Principle', 'K Shot Example'],
            promptsection: ['System_prompt', 'Task_prompt', 'Principle', 'K_shot_example'],
            currentprompt: [
                { "value": 'System_prompt', "text": "", "name": "System Prompt" },
                { "value": 'Task_prompt', "text": "", "name": "Task Prompt" },
                { "value": 'Principle', "text": "", "name": "Principle" },
                { "value": 'K_shot_example', "text": "", "name": "K Shot Example" },
            ],
            activeNames: ['System Prompt', 'Task Prompt'],
            isCollapseMode: true,
            rawprompt: '',
            templateData: [],
            selected: null,
            myprinciple: "",
            mykshot: "",
            current: []
        };
    },
    mounted() {
        // this.getCurrentPrompt()
    },
    watch: {
        uploadedPrinciple(v) {
            for (let i = v.length - 1; i >= 0; i--) {
                if (!this.currentprompt[2].text.includes(v[i])){
                    this.currentprompt[2].text.unshift(v[i])
                }
            }
        },
        uploadedKshot(v) {
            this.currentprompt[3].text = {}
            Object.keys(v).forEach(key => {
                this.currentprompt[3].text[key] = v[key].reasoning.explanation
            })
        },
        selected(v) {
            const index = Number(v[v.length - 1])
            this.currentprompt = this.templateData[index]
        },
        initial(v){
            console.log("initial", v)
            this.getCurrentPrompt()
        }
    },
    methods: {
        isObject(val) {
            return val !== null && typeof val === 'object' && !Array.isArray(val);
        },
        submitHistory(v) {
            this.$emit('currentp', v);
        },
        objectsAreEqual(obj1, obj2) {
            const keys1 = Object.keys(obj1).sort();
            const keys2 = Object.keys(obj2).sort();
            if (keys1.length !== keys2.length) {
                return false;
            }
            for (let i = 0; i < keys1.length; i++) {
                const key = keys1[i];
                if (key !== keys2[i] || obj1[key] !== obj2[key]) {
                    if (typeof obj1[key] === 'object') {
                        if (this.objectsAreEqual(obj1[key], obj2[key])) {
                            continue
                        }
                    } else {
                        return false;
                    }
                }
            }
            return true;
        },
        async getTemplate() {
            var results = await requesthelp.axiosGet("/load_prompt_template");
            console.log("structerd prompt", results["prompt_template"])
            this.templateData = []
            // this.selected = null
            for (let i = 0; i < results["prompt_template"].length; i++) {
                var onetemplate = []
                for (let key in this.promptsection) {
                    var template = {}
                    template.value = this.promptsection[key]
                    template.text = results["prompt_template"][i][this.promptsection[key]]
                    template.name = this.promptname[key]
                    onetemplate.push(template)
                }
                if (this.objectsAreEqual(results["prompt_template"][i], this.prompt)) {
                    this.selected = 'radio' + i.toString()
                    this.templateData.push(onetemplate)
                } else {
                    this.templateData.push(onetemplate)
                }
            }
        },
        async getCurrentPrompt() {
            this.prompt = this.initial["initial_prompt"]

            for (let i = 0; i < this.promptsection.length; i++) {
                let key = this.promptsection[i]
                if (this.prompt[key] != "None") {
                    this.currentprompt[i]["text"] = this.prompt[key]
                    this.rawprompt = this.rawprompt + this.prompt[key]
                }
            }
            this.current = JSON.parse(JSON.stringify(this.currentprompt));
            this.submitHistory(this.currentprompt)
            this.getTemplate()
        },
        edit() {
            console.log("edit button")
        },
        submit() {
            // console.log("submit button")
            if (this.myprinciple !== "") {
                this.currentprompt[2].text.push(this.myprinciple)
            }
            // this.currentprompt[3].text.push(this.mykshot)
            this.myprinciple = ""
            // this.mykshot = ""
            // this.submitHistory(this.currentprompt)

            if (JSON.stringify(this.current) !== JSON.stringify(this.currentprompt)){
                this.current = JSON.parse(JSON.stringify(this.currentprompt));
                console.log(this.current)
                this.$emit('currentp', this.current);
            }

            for (let i = this.currentprompt[2].text.length - 1; i >= 0; i--) {
                if (this.currentprompt[2].text[i] === "") {
                    this.currentprompt[2].text.splice(i, 1);
                }
            }
        },
        handleSwitchChange() {
            console.log("switchChange")
        }
    }

};
</script>

<style>
.promptcontainer {
    height: 89%;
    overflow-y: auto;
}

.selections {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-left: 10px;
    font-size: 16px;
}

.currentpromptdiv {
    height: 93%;
    width: 100%;
}

.selectTitle {
    padding-left: 10px;
    font-size: 20px;
    font-family: sans-serif;;
    padding-bottom: 10px;
    font-weight: 600;
}

.currentPrompt {
    /* background-color: lightgray; */
    height: 85%;
    width: 100%;
    /* overflow: auto; */
    padding-bottom: 10px;
}

.info {
    height: 7%;
    padding-bottom: 5px;
}


.el-switch {
    padding: 5px 5px;
    line-height: 16px !important;
    height: 16px !important;
}

.el-select {
    box-sizing: border-box;
    padding: 2px 12px;
    height: 24px !important;
}

.el-select__wrapper {
    display: flex;
    align-items: center;
    position: relative;
    box-sizing: border-box;
    cursor: pointer;
    text-align: left;
    font-size: 20;
    padding: 2 4;
    gap: 0;
    min-height: 20px !important;
    line-height: 20px !important;
    border-radius: var(--el-border-radius-base);
    background-color: var(--el-fill-color-blank);
    transition: var(--el-transition-duration);
    box-shadow: 0 0 0 1px var(--el-border-color) inset;
}

.el-radio {
    margin-right: 3px !important;
    height: auto !important;
}

.el-collapse-item__header {
    font-size: 16px !important;
    height: 30px !important;
    line-height: 24px !important;
}

.el-collapse-item__content {
    font-size:10px;
    padding-bottom: 5px !important;
}

.rawinput {
    /* min-height: 85%; */
    height: 100%;
}

.el-textarea {
    width: 95% !important;
}

.rawinput .el-textarea__inner {
    height: 100%;
    /* This will make the textarea fill the height of its parent */
    padding-left: 10px;
    padding-right: 10px;
    width: 100%
}

.promptinput .el-textarea__inner {
    /* height: 80px; */
    /* height: auto; */
    font-family: sans-serif;
    font-size: 15px;
    padding-left: 10px;
    padding-right: 10px;
    width: 100%;
    max-height: 100px !important;
    /* min-height: 100px !important;  */
}

.promptButtons {
    display: flex;
    justify-content: flex-start;
    margin-left: 10px;
}

.switchbutton {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 10px;
    font-size: 15px;
    font-weight: 300;
    font-family: sans-serif;
    height: 10%;
}

.modelPerformance {
    width: 100%;
}

.currentPrompt .el-textarea .el-textarea__inner {
    border-radius: 4px;
    border: 1px solid #dcdfe6;
    /* Add more styles as per your requirement */
}

.promptpanelContainer {
    height: 37vh;
    width: 100%;
}

.popuptemplate {
    margin-left: 20px;
}

.mywarp {
    white-space: normal;
    overflow-wrap: break-word;
}

.el-button {
    height: 30px !important;
}
</style>
