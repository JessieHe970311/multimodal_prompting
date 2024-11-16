<template>
    <div class="prompthistoryContainer viewbottom view">
        <div class="history">
            <div class="selectTitle" style="height: 5%">Prompt History</div>

            <!-- <div v-html="diffAsHtml"></div> -->
            <div class="historyView">
                <!-- <el-collapse v-model="activeNames">
                    <el-collapse-item
                    v-for="item in history_prompt"
                    :key="item.name"
                    :name="item.name"
                    :title="item.name"
                    class="collapse-prompt-custom"
                    >
                    <div>{{ item.prompt_text }}</div>
                    </el-collapse-item>
                </el-collapse> -->
                <div v-for="(historyGroup, index) in richtextHistoryGroups" :key="index">
                    <el-collapse v-model="activeoutNames">
                        <el-collapse-item v-model="richtextHistoryGroups[index]"
                            class="collapse-prompt-custom inner-collapse" :name="richtextHistoryGroups.length - index">
                            <template #title>
                                Version {{ richtextHistoryGroups.length - index }} &nbsp;
                                &nbsp;
                                Acc: {{ historyGroup.acc}} &nbsp;
                                <!-- {{ historyGroup[0].change }}{{ historyGroup[1].change }}{{ historyGroup[2].change }}{{ historyGroup[3].change }} -->
                                System:  &nbsp;
                                <font-awesome-icon v-if="historyGroup[0].change === '-'" :icon="['far', 'square-minus']" style="color: #957fef;" />
                                <font-awesome-icon v-if="historyGroup[0].change === '+'" :icon="['far', 'square-plus']" style="color: #D59534;" />
                                <font-awesome-icon v-if="historyGroup[0].change === '0'" :icon="['far', 'square']" style="color: #ffe272;" />
                                <font-awesome-icon v-if="historyGroup[0].change === '/'" :icon="['far', 'square']" style="color: #c0c0c0;" />
                                &nbsp;
                                Task:  &nbsp;
                                <font-awesome-icon v-if="historyGroup[1].change === '-'" :icon="['far', 'square-minus']" style="color: #957fef;" />
                                <font-awesome-icon v-if="historyGroup[1].change === '+'" :icon="['far', 'square-plus']" style="color: #D59534;" />
                                <font-awesome-icon v-if="historyGroup[1].change === '0'" :icon="['far', 'square']" style="color: #ffe272;" />
                                <font-awesome-icon v-if="historyGroup[1].change === '/'" :icon="['far', 'square']" style="color: #c0c0c0;" />
                                &nbsp;
                                Principle:  &nbsp;
                                <font-awesome-icon v-if="historyGroup[2].change === '-'" :icon="['far', 'square-minus']" style="color: #957fef;" />
                                <font-awesome-icon v-if="historyGroup[2].change === '+'" :icon="['far', 'square-plus']" style="color: #D59534;" />
                                <font-awesome-icon v-if="historyGroup[2].change === '0'" :icon="['far', 'square']" style="color: #ffe272;" />
                                <font-awesome-icon v-if="historyGroup[2].change === '/'" :icon="['far', 'square']" style="color: #c0c0c0;" />
                                &nbsp;
                                K-shot  &nbsp;
                                <font-awesome-icon v-if="historyGroup[3].change === '-'" :icon="['far', 'square-minus']" style="color: #957fef;" />
                                <font-awesome-icon v-if="historyGroup[3].change === '+'" :icon="['far', 'square-plus']" style="color:  #D59534;" />
                                <font-awesome-icon v-if="historyGroup[3].change === '0'" :icon="['far', 'square']" style="color: #ffe272;" />
                                <font-awesome-icon v-if="historyGroup[3].change === '/'" :icon="['far', 'square']" style="color: #c0c0c0;" />
                            </template>
                            <el-collapse-item v-for="item in historyGroup" :key="item.value" :name="richtextHistoryGroups.length - index +item.name"
                                class="collapse-prompt-custom inner-collapse">
                                <template #title>
                                    <div>
                                        {{item.name}}
                                        <font-awesome-icon v-if="item.change === '-'" :icon="['far', 'square-minus']" style="color: #957fef;" />
                                        <font-awesome-icon v-if="item.change === '+'" :icon="['far', 'square-plus']" style="color: #D59534;" />
                                        <font-awesome-icon v-if="item.change === '0'" :icon="['far', 'square']" style="color: #ffe272;" />
                                        <font-awesome-icon v-if="item.change === '/'" :icon="['far', 'square']" style="color: #c0c0c0;" />
                                    </div>
                                </template>
                                <div v-if="Array.isArray(item.text)">
                                    <div v-for="(text, index) in item.text" :key="index">
                                        <!-- <div style="padding: 10px; text-align:left;" v-html="item.text[index]">
                                        </div> -->
                                        <el-card style="padding: 2px; text-align: left; font-size:inherit; font-family:inherit;">
                                            <div v-html="item.text[index]"></div>
                                        </el-card>s
                                    </div>
                                </div>

                                <div v-else-if="isObject(item.text)">
                                    <!-- If item.text is an object, display each key-value pair -->
                                    <div v-for="(value, key) in item.text" :key="key">
                                        <label>{{ key }}:</label>
                                        <div style="padding: 10px; text-align:left; font-size:inherit; font-family:inherit;" v-html="item.text[key]">
                                        </div>
                                    </div>
                                </div>

                                <div v-else style="padding: 10px; text-align:left; font-size:inherit; font-family:inherit;" v-html="item.text">
                                </div>
                            </el-collapse-item>
                        </el-collapse-item>
                    </el-collapse>
                </div>
            </div>
            <div class="historyAccView">
                <svg id="accsvg" style="width: 100%; height: 92%">

                </svg>
            </div>
        </div>
    </div>
</template>

<script setup>
import {
    ElCollapse,
    ElCollapseItem,
    ElCard
} from 'element-plus'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faSquareMinus, faSquarePlus, faSquare } from '@fortawesome/free-regular-svg-icons'
// import { faSquare} from '@fortawesome/free-solid-svg-icons'
</script>

<script>
import requesthelp from "../service/request.js";
import diff_match_patch from 'diff-match-patch';
import * as d3 from 'd3';

export default {
    props: ['lastPrompt','kshotExampleData','allInstancedata'],
    data() {
        return {
            activeNames: [],
            activeoutNames: [],
            history_prompt: [],
            diffAsHtml: '',
            historyGroups: [],
            promptsection: ['System_prompt', 'Task_prompt', 'Principle', 'K_shot_example'],
            richtextHistoryGroups: [],
            kshotall: {},
            newresults: {},
            accRecord:[]
        };
    },
    components: {
        'font-awesome-icon': FontAwesomeIcon
    },
    created() {
        library.add(faSquareMinus,faSquarePlus,faSquare)
    },
    mounted() {
        // this.getHistoryPrompts()
        // this.computeDiff("hello world", "goodbye world")
    },
    watch: {
        lastPrompt: {
            async handler(v) {
                console.log("history",v)
                if (this.historyGroups.length === 0) {
                    var current = JSON.parse(JSON.stringify(v));
                    // v.acc = this.allInstancedata["initial_model_performance"]["acc"]
                    this.historyGroups.unshift(current)
                    // this.historyGroups[0].acc = this.allInstancedata["initial_model_performance"]["acc"].toFixed(2) 
                    this.richtextHistoryGroups.unshift(current)
                    this.richtextHistoryGroups[0].acc = this.allInstancedata["initial_model_performance"]["acc"].toFixed(2) 
                    this.accRecord.push({time:this.accRecord.length+1, value:parseFloat(this.richtextHistoryGroups[0].acc)})
                    this.richtextHistoryGroups[0][0].change = "/"
                    this.richtextHistoryGroups[0][1].change = "/"
                    this.richtextHistoryGroups[0][2].change = "/"
                    this.richtextHistoryGroups[0][3].change = "/"
                } else {
                    current = JSON.parse(JSON.stringify(v));
                    var last = this.historyGroups[0]

                    for (let i = 0; i < current.length; i++) {
                        if (current[i].value === this.promptsection[0] || current[i].value === this.promptsection[1]) {
                            if (current[i].text.length > last[i].text.length){
                                current[i].change = "+"
                            }else if (current[i].text.length < last[i].text.length){
                                current[i].change = "-"
                            }else{
                                if (last[i].text === current[i].text){
                                    current[i].change = "/"
                                }else{
                                    current[i].change = "0"
                                }
                            }
                            current[i].text = this.computeDiff(last[i].text, current[i].text)
                        } else if (current[i].value === this.promptsection[2]) {
                            for (let j = last[i].text.length - 1; j >= 0; j--) {
                                if (last[i].text[j] === "") {
                                    last[i].text.splice(j, 1);
                                }
                            }
                            var validLength = 0
                            for (let l = 0; l<current[i].text.length;l++){
                                if (current[i].text[l] !== ""){
                                    validLength = validLength + 1
                                }
                            }
                            // var validLengthLast = 0
                            // for (let l = 0; l<current[i].text.length;l++){
                            //     if (current[i].text[l] !== ""){
                            //         validLengthLast = validLengthLast + 1
                            //     }
                            // }
                            if (validLength > last[i].text.length){
                                current[i].change = "+"
                            }else if (validLength < last[i].text.length){
                                current[i].change = "-"
                            }else{
                                current[i].change = "/"
                                for (let j = 0; j < current[i].text.length; j++){
                                    if (current[i].text[j] !== last[i].text[j]){
                                        current[i].change = "0"
                                        break
                                    }else{
                                        current[i].change = "/"
                                    }
                                }
                            }
                            for (let j = 0; j < current[i].text.length; j++) {
                                if (j < last[i].text.length){
                                    current[i].text[j] = this.computeDiff(last[i].text[j], current[i].text[j])
                                }else{
                                    current[i].text[j] = `<ins style="background-color: #EACA99;">${current[i].text[j]}</ins>`
                                }
                            }

                        } else if (current[i].value === this.promptsection[3]) {
                            if (Object.keys(current[i].text).length > Object.keys(last[i].text).length){
                                current[i].change = "+"
                            }else if (current[i].text.length < last[i].text.length){
                                current[i].change = "-"
                            }else{
                                current[i].change = "/"
                                Object.keys(current[i].text).forEach(key => {
                                    if (current[i].text[key] !== last[i].text[key]){
                                        current[i].change = "0"
                                    }
                                })
                            }
                            Object.keys(current[i].text).forEach(key => {
                                if (key in last[i].text) {
                                    current[i].text[key] = this.computeDiff(last[i].text[key], current[i].text[key])
                                } else {
                                    current[i].text[key] = `<ins style="background-color: #EACA99;">${current[i].text[key]}</ins>`
                                }
                            });
                        }
                    }
                    this.richtextHistoryGroups.unshift(current)
                    this.historyGroups.unshift(v)
                    this.richtextHistoryGroups[0].acc = (await this.getAccuracy(v)).toFixed(2)
                    this.accRecord.push({time:this.accRecord.length+1, value:parseFloat(this.richtextHistoryGroups[0].acc)})
                }
            },
            deep: true
        },
        accRecord:{
            async handler(v) {
                this.drawAcc()
            },
            deep:true
        },
        kshotExampleData(v){
            this.kshotall = v["k_shot_example_dict"]
        }
    },
    methods: {
        drawAcc(){
            const svgElement = document.getElementById("accsvg");
            const width = svgElement.clientWidth;
            const height = svgElement.clientHeight;
            const margin = { top: 5, right: 0, bottom: 10, left: 30 };
            // const svgwidth = width - margin.left - margin.right;
            const singleWidth = 0.1 * width;
            const svgwidth = (this.accRecord.length+1) * singleWidth;
            const svgheight = height - margin.top - margin.bottom
            d3.select("#accsvg").selectAll("*").remove();
            var svg = d3.select("#accsvg")
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            const lineData = this.accRecord
            var y = d3.scaleLinear()
                .domain( [0, 1])
                .range([ svgheight, 0 ]);
            
            svg.append("g")
                .attr("class", "axis")
                .call(d3.axisLeft(y)
                        .ticks(3)
                        .tickSize(-svgwidth) // 刻度线延伸到整个图表宽度的内部
                )
                .selectAll(".tick line") // 选择所有的刻度线
                .style("stroke", "#b3b3b3"); // 设置刻度线颜色为淡灰色

            // svg.append("line")      // Append a new line element
            //     .attr("x1", 5)                  // x position of the start of the line
            //     .attr("y1", y(0.5))                  // y position of the start of the line
            //     .attr("x2", svgwidth)                 // x position of the end of the line
            //     .attr("y2", y(0.5))                  // y position of the end of the line
            //     .style("stroke", "#d9d9d9")        // Line color
            //     .style("stroke-width", 1); 
            // svg.append("line")      // Append a new line element
            //     .attr("x1", 5)                  // x position of the start of the line
            //     .attr("y1", y(1))                  // y position of the start of the line
            //     .attr("x2", svgwidth)                 // x position of the end of the line
            //     .attr("y2", y(1))                  // y position of the end of the line
            //     .style("stroke", "#d9d9d9")        // Line color
            //     .style("stroke-width", 1); 
            // svg.append("text")               // Append a new text element
            //     .attr("x", 0)
            //     .attr("y", y(0.5))
            //     .attr("fill", "black")        // Text color
            //     .style("font-size", "12px") 
            //     .style("dominant-baseline", "middle") 
            //     .style("text-anchor", "end")  // Font size
            //     .attr("padding-right","2px")
            //     .text("0.5");
            // svg.append("text")               // Append a new text element
            //     .attr("x", 0)
            //     .attr("y", y(1))
            //     .attr("fill", "black")        // Text color
            //     .style("font-size", "12px")
            //     .style("dominant-baseline", "middle")  
            //     .style("text-anchor", "end")   // Font size
            //     .attr("padding","2px")
            //     .text("1");

            // svg.append("line")      // Append a new line element
            //     .attr("x1", 5)                  // x position of the start of the line
            //     .attr("y1", svgheight)                  // y position of the start of the line
            //     .attr("x2", svgwidth)                 // x position of the end of the line
            //     .attr("y2", svgheight)                  // y position of the end of the line
            //     .style("stroke", "black")        // Line color
            //     .style("stroke-width", 1); 

            lineData.forEach(lineData => {
                svg.append("path")
                    .datum(lineData)
                    .attr("fill", "none")
                    .attr("stroke", "#5174bb")
                    .attr("stroke-width", 1.5)
                    .attr("d", d3.line()
                        .x(function(d) { return (d.time * singleWidth) })
                        .y(function(d) { return y(d.value) })
                        )
            });

            svg.append("path")
                .datum(lineData)
                .attr("fill", "none")
                .attr("stroke", "#5174bb")
                .attr("stroke-width", 3)
                .attr("d", d3.line()
                    .x(function(d) { return (d.time * singleWidth - singleWidth*0.5) })
                    .y(function(d) { return y(d.value) })
                    )
        //     // Add the points
            svg
                .append("g")
                .selectAll("dot")
                .data(lineData)
                .enter()
                .append("circle")
                    .attr("cx", function(d) { return (d.time * singleWidth - singleWidth*0.5) } )
                    .attr("cy", function(d) { return y(d.value) } )
                    .attr("r", 4)
                    .attr("fill", "white")
                    .attr("stroke", "#5174bb")
                    .attr("stroke-width", "3px")
            
            svg.append("g")
                .selectAll("text")
                .data(lineData)
                .enter()
                .append("text")               // Append a new text element
                .attr("x", function(d) { return (d.time * singleWidth - singleWidth*0.5) } )
                .attr("y", function(d) { return (svgheight+margin.bottom) } )
                .attr("fill", "black")        // Text color
                .style("font-size", "12px")   // Font size
                .style("text-anchor", "middle")
                .text(function(d) { return ("v" + d.time)});
            
        },
        async getAccuracy(v){
            var myObject = {}
            for (let i = 0; i < v.length -1; i++) {
                myObject[v[i].value] = v[i].text
            }
            var mylist = []
            var mydict = {}
            Object.keys(v[3].text).forEach(key => {
                mylist.push(key)
                this.kshotall[key].reasoning.explanation = v[3].text[key]
                mydict[key] = this.kshotall[key]
            })
            myObject[v[3].value] = {
                k_shot_example_dict: mydict,
                k_shot_example_list: mylist
            }

            // return 0.7
            console.log("upload prompt", myObject)
            console.log("loading")
            var results = await requesthelp.axiosPost("load_prompt_and_result", {
                prompt: myObject
            })
            console.log("get results", results)
            this.newresults = results
            console.log(results["model_performance"]["acc"])
            this.$emit('newresults', this.newresults);
            return results["model_performance"]["acc"]

        },
        isObject(val) {
            return val !== null && typeof val === 'object' && !Array.isArray(val);
        },
        async getHistoryPrompts() {
            var results = await requesthelp.axiosGet("/get_history_prompt")
            this.history_prompt = results;
            // console.log("history_prompt",this.history_prompt)
        },
        computeDiff(text1, text2) {
            const dmp = new diff_match_patch();
            const diffs = dmp.diff_main(text1, text2);
            dmp.diff_cleanupSemantic(diffs);
            const html = this.diffToHtml(diffs);
            // this.diffAsHtml = html;
            return html
        },
        diffToHtml(diffs) {
            const html = diffs.map(([operation, text]) => {
                text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                if (operation === 0) { // No change
                    return `<span>${text}</span>`;
                }
                if (operation === -1) { // Deletion
                    return `<del style="background-color: #BCBAD6;">${text}</del>`;
                }
                if (operation === 1) { // Insertion
                    // return `<ins style="background-color: #74c0fc80;">${text}</ins>`;
                    return `<ins style="background-color: #EACA99;">${text}</ins>`;
                }
            }).join('');

            return html;
        }
    }
}

</script>

<style>
.prompthistoryContainer {
    height: 40vh;
}

.history {
    height: 100%;
}

.historyView {
    /* background-color: lightgray; */
    height: 68%;
    overflow-y: auto;
}

.historyAccView {
    /* background-color: lightgray; */
    height: 25%;
    overflow-x: scroll;
    overflow-y: hidden;
}

.collapse-prompt-custom .el-collapse-item__header {
    padding-left: 5px;
    cursor: pointer;
}

.el-collapse {
    margin-left: 10px;
    font-family: sans-serif;

}

.inner-collapse {
    margin-left: 20px;
}

.el-card__body {
    padding: 5px !important;
}
</style>