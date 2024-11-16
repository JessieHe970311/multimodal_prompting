<template>
    <div class="reasoningpathContainer viewbottom view">
        <!-- <div class="selectTitle">Modality Level/Reasoning path</div> -->
        <div class="reasoningpathView">
            <div class="pathleft">
                <el-button @click="toggleBrush" style="padding-top:10px">Toggle Brush/Hover</el-button>
                <svg id="pathsvg" style="width: 100%; height: 94%">

                </svg>
            </div>
            <div class="pathright">
                <el-table class="righttable" :data="myTable" style="width: 100%" @sort-change="renderAll"
                    @cell-click="handleCellclick" :cell-class-name="getCellClassName" @expand-change="onTableRowExpand">
                    <el-table-column type="expand" width="35px" fixed>
                        <template v-slot="scope">
                            <!-- <svg :ref="`svg${scope.row.initial_index}`" :id="`svg${scope.row.initial_index}`" style="width:calc(100% - 205px); height: 150px; margin-left:35px;"> -->
                            <!-- </svg> -->
                            <div
                                style="margin-left:35px; margin-right:170px; width:calc(100%-205px); display: flex; align-items: center;">
                                <div style="width:50%;">
                                    <svg v-if="scope.row.language.length !== 0"
                                        :ref="`lansvg${scope.row.initial_index}`"
                                        :id="`lansvg${scope.row.initial_index}`" style="width:100%; height: 150px">
                                    </svg>
                                    <el-pagination v-if="scope.row.language.length >= 2" layout="prev, pager, next"
                                        :total="scope.row.language.length * 10"
                                        @current-change="newPage => handleCurrentChangeLan(newPage, scope.row.initial_index)" />
                                </div>
                                <div style="width:50%;">
                                    <svg v-if="scope.row.visual.length !== 0" :ref="`vissvg${scope.row.initial_index}`"
                                        :id="`vissvg${scope.row.initial_index}`" style="width:100%; height: 150px">
                                    </svg>
                                    <el-pagination v-if="scope.row.visual.length >= 2" layout="prev, pager, next"
                                        :total="scope.row.visual.length * 10"
                                        @current-change="newPage => handleCurrentChangeVis(newPage, scope.row.initial_index)" />
                                </div>
                                <!-- <div style="width:30%;">
                                    <img src="../assets/images/legend-02.png" style="width: 60%;margin-left:20%">
                                </div> -->
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column label="Language" class-name="border-right" fixed>
                        <template v-slot="scope">
                            <div v-for="item  in scope.row.language" :key="item.key" style="display: flex;">
                                <div style="width: 60%"> {{ item.represent }} </div>
                                <svg :ref="`${item.key}`" :id="`${item.key}`" style="width: 40%; height: 30px">
                                </svg>
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column label="Visual" class-name="border-right border-left" fixed>
                        <template v-slot="scope">
                            <div v-for="item  in scope.row.visual" :key="item.key" style="display: flex;">
                                <div style="width: 60%"> {{ item.represent }} </div>
                                <svg :ref="`${item.key}`" :id="`${item.key}`" style="width: 40%; height: 30px">
                                </svg>
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column prop="instance_number" label="Total" sortable width="85px" fixed />
                    <el-table-column prop="error_number" label="Error" sortable width="85px" fixed />
                </el-table>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ElTable, ElTableColumn, ElButton, ElPagination } from "element-plus";
</script>

<script>
import requesthelp from "../service/request.js";
import * as d3 from 'd3';
import cloud from 'd3-cloud';

export default {
    props:['allInstancedata'],
    data() {
        return {
            allData: {},
            myTable: [],
            allInstanceData: {},

            // const colorarray = ["#c8e6c9", "#fff3d0", "#ffcdd2"] //红绿灯版
            // const colorarray = ["#CB9E9E", "#fff3d0", "#919FC3"] //暗淡无光版
            // colorarray: ["#E7AAAA", "#fff3d0", "#B2C1E8"],
            // 顺序：pos neu neg
            // colorarray: ["#CDA2A2", "#F5EEE3", "#A1AECE"],
            // colorarray: ["#92413A", "#F5EEE3", "#7D92B5"],
            // colorarray: ["#BF675F", "#F5EEE3", "#7D92B5"],
            colorarray: ["#CB827C", "#F4EDE1", "#8A9DBC"],
            // colorarray: ["#A04840", "#F5EEE3", "#7D92B5"],
            // colorarray: ["#B95950", "#F5EEE3", "#7D92B5"],
            colorarrayrgb: [d3.rgb(203, 130, 124), d3.rgb(244, 237, 225), d3.rgb(138, 157, 188)],
            // errorcolor: ["#BBABD2", "#BCD79A"]
            // 顺序：false true
            errorcolor: ["#CCBEB8", "#DADCD0"]
        };
    },
    mounted() {
        // this.getData()
        // this.allInstanceData = this.allInstancedata
    },
    watch:{
        allInstancedata(v){
            // console.log(v)
            this.allInstanceData = v["initial_result"];
            this.allData = v["frequent_set_data"]["frequent_itemsets"]
            this.fetchData()
            this.renderAll()
            this.renderPath()
        }
    },
    methods: {
        async getData() {
            var results = await requesthelp.axiosGet("/load_initial_prompt_and_result")
            this.allInstanceData = results["initial_result"];
            // results = await requesthelp.axiosGet("/load_data_for_center_view_table")
            console.log("center data", results["frequent_set_data"])
            console.log("instance data", this.allInstanceData)
            this.allData = results["frequent_set_data"]["frequent_itemsets"]
            this.fetchData()
            this.renderAll()
            this.renderPath()
        },
        renderAll() {
            this.renderRatio()
            // this.renderWordcloud()
        },
        renderPath() {
            // dataset inspection
            for (let key in this.allInstanceData) {
                if (this.allInstanceData[key].interaction_type === "R" && this.allInstanceData[key].modality_type !== "complement") {
                    console.log("data inspection", this.allInstanceData[key])
                }
            }

            // re-structured dataset
            let realldata = []
            for (let key in this.allInstanceData) {
                if (this.allInstanceData[key].prediction) {
                    for (let predKey in this.allInstanceData[key].prediction) {
                        let value = this.allInstanceData[key].prediction[predKey];
                        if (typeof value === 'string') {
                            this.allInstanceData[key].prediction[predKey] = value.toLowerCase();
                        }
                    }
                }

                var oneinstance = { "videoname": key }
                realldata.push({ ...oneinstance, ...this.allInstanceData[key] })
            }
            console.log("restructured all data", realldata)

            // 确认是不是只有pos neg neutral
            const all_prediction_type = Array.from(new Set(realldata.map(item => item.prediction.overall)));
            if (all_prediction_type.length > 3) {
                console.log("do not have correct prediction types: ", all_prediction_type)
            } else {
                console.log("prediction type: ", all_prediction_type)
            }
            const all_prediction_type_language = Array.from(new Set(realldata.map(item => item.prediction.language)));
            if (all_prediction_type_language.length > 3) {
                console.log("do not have correct language prediction types: ", all_prediction_type_language)
            } else {
                console.log("prediction type: ", all_prediction_type_language)
            }
            const all_prediction_type_visual = Array.from(new Set(realldata.map(item => item.prediction.visual)));
            if (all_prediction_type_visual.length > 3) {
                console.log("do not have correct visual prediction types: ", all_prediction_type_visual)
            } else {
                console.log("prediction type: ", all_prediction_type_visual)
            }

            // rank dataset === five dataset
            const priority = {
                positive: 1,
                neutral: 2,
                negative: 3
            };

            const modalityOrder = ["complement", "conflict"];
            // const interactionOrder = ["R", "U1", "U2", "S", "undecided"];
            const interactionOrder = ["R", "complement-distinct", "conflict-dominant", "conflict-distinct"];
            const modalityPredictionOrder = ["positive", "neutral", "negative"]

            // first row dataset
            const firstRowData = [...realldata].sort((a, b) => {
                let overallA = a.prediction ? a.prediction.overall : null;
                let overallB = b.prediction ? b.prediction.overall : null;
                if ((priority[overallA] || 0) !== (priority[overallB] || 0)) {
                    return (priority[overallA] || 0) - (priority[overallB] || 0);
                }
                let modalityA = a.modality_type;
                let modalityB = b.modality_type;
                let modalityIndexA = modalityOrder.indexOf(modalityA);
                let modalityIndexB = modalityOrder.indexOf(modalityB);
                modalityIndexA = modalityIndexA === -1 ? modalityOrder.length : modalityIndexA;
                modalityIndexB = modalityIndexB === -1 ? modalityOrder.length : modalityIndexB;
                if (modalityIndexA - modalityIndexB === 0) {
                    let interactionA = a.interaction_type;
                    let interactionB = b.interaction_type;
                    let interactionIndexA = interactionOrder.indexOf(interactionA);
                    let interactionIndexB = interactionOrder.indexOf(interactionB);
                    interactionIndexA = interactionIndexA === -1 ? interactionOrder.length : interactionIndexA;
                    interactionIndexB = interactionIndexB === -1 ? interactionOrder.length : interactionIndexB;
                    if (interactionIndexA === interactionIndexB){
                        let languageA = a.prediction.language.toLowerCase()
                        let languageB = b.prediction.language.toLowerCase()
                        let languageIndexA = modalityPredictionOrder.indexOf(languageA);
                        let languageIndexB = modalityPredictionOrder.indexOf(languageB);
                        languageIndexA = languageIndexA === -1 ? modalityPredictionOrder.length : languageIndexA;
                        languageIndexB = languageIndexB === -1 ? modalityPredictionOrder.length : languageIndexB;
                        if (languageIndexA === languageIndexB){
                            let visualA = a.prediction.visual.toLowerCase()
                            let visualB = b.prediction.visual.toLowerCase()
                            let visualIndexA = modalityPredictionOrder.indexOf(visualA);
                            let visualIndexB = modalityPredictionOrder.indexOf(visualB);
                            visualIndexA = visualIndexA === -1 ? modalityPredictionOrder.length : visualIndexA;
                            visualIndexB = visualIndexB === -1 ? modalityPredictionOrder.length : visualIndexB;
                            return visualIndexA - visualIndexB;
                        }
                        return languageIndexA - languageIndexB;
                    }
                    return interactionIndexA - interactionIndexB;
                }
                return modalityIndexA - modalityIndexB;
            });
            let counts = realldata.reduce((v, { prediction }) => ({
                positive: v.positive + (prediction && prediction.overall === 'positive' ? 1 : 0),
                negative: v.negative + (prediction && prediction.overall === 'negative' ? 1 : 0),
                neutral: v.neutral + (prediction && prediction.overall === 'neutral' ? 1 : 0),
                all: v.all + 1
            }), { positive: 0, neutral: 0, negative: 0, all: 0 });
            const firstRowCountData = { ...{ "count": counts }, ...{ "detail": firstRowData } }
            console.log("first row data", firstRowCountData)

            // second row dataset
            const all_modality_type = Array.from(new Set(realldata.map(item => item.modality_type)));
            console.log("all modality types: ", all_modality_type)
            var secondRowData = {}
            var secondRowCountData = {}
            for (let key in all_modality_type) {
                secondRowData[all_modality_type[key]] = realldata.filter(item => item.modality_type === all_modality_type[key]);
                secondRowData[all_modality_type[key]].sort((a, b) => {
                    let overallA = a.prediction ? a.prediction.overall : null;
                    let overallB = b.prediction ? b.prediction.overall : null;
                    if ((priority[overallA] || 0) !== (priority[overallB] || 0)) {
                        return (priority[overallA] || 0) - (priority[overallB] || 0);
                    }
                    let interactionA = a.interaction_type;
                    let interactionB = b.interaction_type;
                    let interactionIndexA = interactionOrder.indexOf(interactionA);
                    let interactionIndexB = interactionOrder.indexOf(interactionB);
                    interactionIndexA = interactionIndexA === -1 ? interactionOrder.length : interactionIndexA;
                    interactionIndexB = interactionIndexB === -1 ? interactionOrder.length : interactionIndexB;
                    if (interactionIndexA === interactionIndexB){
                        let languageA = a.prediction.language.toLowerCase()
                        let languageB = b.prediction.language.toLowerCase()
                        let languageIndexA = modalityPredictionOrder.indexOf(languageA);
                        let languageIndexB = modalityPredictionOrder.indexOf(languageB);
                        languageIndexA = languageIndexA === -1 ? modalityPredictionOrder.length : languageIndexA;
                        languageIndexB = languageIndexB === -1 ? modalityPredictionOrder.length : languageIndexB;
                        if (languageIndexA === languageIndexB){
                            let visualA = a.prediction.visual.toLowerCase()
                            let visualB = b.prediction.visual.toLowerCase()
                            let visualIndexA = modalityPredictionOrder.indexOf(visualA);
                            let visualIndexB = modalityPredictionOrder.indexOf(visualB);
                            visualIndexA = visualIndexA === -1 ? modalityPredictionOrder.length : visualIndexA;
                            visualIndexB = visualIndexB === -1 ? modalityPredictionOrder.length : visualIndexB;
                            return visualIndexA - visualIndexB;
                        }
                        return languageIndexA - languageIndexB;
                    }
                    return interactionIndexA - interactionIndexB;
                });
                counts = secondRowData[all_modality_type[key]].reduce((v, { prediction }) => ({
                    positive: v.positive + (prediction && prediction.overall === 'positive' ? 1 : 0),
                    negative: v.negative + (prediction && prediction.overall === 'negative' ? 1 : 0),
                    neutral: v.neutral + (prediction && prediction.overall === 'neutral' ? 1 : 0),
                    all: v.all + 1
                }), { positive: 0, neutral: 0, negative: 0, all: 0 });
                secondRowCountData[all_modality_type[key]] = { ...{ "count": counts }, ...{ "detail": secondRowData[all_modality_type[key]] } }
            }
            console.log("second row data", secondRowCountData)

            // third row dataset
            const all_interaction_type = Array.from(new Set(realldata.map(item => item.interaction_type)));
            console.log("all interaction types: ", all_interaction_type)
            var thirdRowData = {}
            var thirdRowCountData = {}
            for (let key in all_interaction_type) {
                thirdRowData[all_interaction_type[key]] = realldata.filter(item => item.interaction_type === all_interaction_type[key]);
                thirdRowData[all_interaction_type[key]].sort((a, b) => {
                    let overallA = a.prediction ? a.prediction.overall : null;
                    let overallB = b.prediction ? b.prediction.overall : null;
                    // return (priority[overallA] || 0) - (priority[overallB] || 0);

                    if ((priority[overallA] || 0) !== (priority[overallB] || 0)) {
                        return (priority[overallA] || 0) - (priority[overallB] || 0);
                    }
                    // if (interactionIndexA === interactionIndexB){
                    let languageA = a.prediction.language.toLowerCase()
                    let languageB = b.prediction.language.toLowerCase()
                    let languageIndexA = modalityPredictionOrder.indexOf(languageA);
                    let languageIndexB = modalityPredictionOrder.indexOf(languageB);
                    languageIndexA = languageIndexA === -1 ? modalityPredictionOrder.length : languageIndexA;
                    languageIndexB = languageIndexB === -1 ? modalityPredictionOrder.length : languageIndexB;
                    if (languageIndexA === languageIndexB){
                        let visualA = a.prediction.visual.toLowerCase()
                        let visualB = b.prediction.visual.toLowerCase()
                        let visualIndexA = modalityPredictionOrder.indexOf(visualA);
                        let visualIndexB = modalityPredictionOrder.indexOf(visualB);
                        visualIndexA = visualIndexA === -1 ? modalityPredictionOrder.length : visualIndexA;
                        visualIndexB = visualIndexB === -1 ? modalityPredictionOrder.length : visualIndexB;
                        return visualIndexA - visualIndexB;
                    }
                    return languageIndexA - languageIndexB;
                    // }
                });
                counts = thirdRowData[all_interaction_type[key]].reduce((v, { prediction }) => ({
                    positive: v.positive + (prediction && prediction.overall === 'positive' ? 1 : 0),
                    negative: v.negative + (prediction && prediction.overall === 'negative' ? 1 : 0),
                    neutral: v.neutral + (prediction && prediction.overall === 'neutral' ? 1 : 0),
                    all: v.all + 1
                }), { positive: 0, neutral: 0, negative: 0, all: 0 });
                thirdRowCountData[all_interaction_type[key]] = { ...{ "count": counts }, ...{ "detail": thirdRowData[all_interaction_type[key]] } }
            }
            console.log("third row data", thirdRowCountData)

            // draw sankey
            const svgElement = document.getElementById("pathsvg");

            const width = svgElement.clientWidth;
            const height = svgElement.clientHeight;

            // const margin = { top: 20, right: 10, bottom: 2, left: 2 };
            const margin = { top: 20, right: 10, bottom: 20, left: 2 };
            const svgwidth = width - margin.left - margin.right;
            const svgheight = height - margin.top - margin.bottom;


            const legend_width = 0.025 * svgwidth

            d3.select("#pathsvg").selectAll("*").remove();

            var legend_svg = d3.select("#pathsvg")
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            var svg = d3.select("#pathsvg")
                .append("g")
                .attr("transform", "translate(" + (margin.left + legend_width) + "," + margin.top + ")");

            const mainbar_type = ["positive", "neutral", "negative"]
            const firstrow_type = ["all"]
            const secondrow_type = ["complement", "conflict"]
            // const thirdrow_type = ["R", "U1", "U2", "S", "undecided"]
            const thirdrow_type = ["R", "complement-distinct", "conflict-dominant", "conflict-distinct"];
            const thirdrow_name = ["Redundant", "Distinct", "Dominant", "Distinct"];
            const namechange = d3.scaleOrdinal()
                                .domain(thirdrow_type)
                                .range(thirdrow_name);

            const padding = svgwidth * 0.02;
            const padding_number = thirdrow_type.length + 1;
            const row_array = [1, 2, 3];

            // const colorarray = ["#c8e6c9", "#fff3d0", "#ffcdd2"] //红绿灯版
            // const colorarray = ["#CB9E9E", "#fff3d0", "#919FC3"] //暗淡无光版
            const colorarray = this.colorarray
            const defaultColor = "#D9D9D9";

            const color = d3.scaleOrdinal()
                .domain(mainbar_type)
                .range(colorarray)
                .unknown(defaultColor);
            // const errorcolor = d3.scaleOrdinal()
            //     .domain([-1, 0, 1])
            //     .range(colorarray)
            //     .unknown(defaultColor);     

            const errorcolor = d3.scaleOrdinal()
                .domain([false, true])
                .range(this.errorcolor)
                .unknown(defaultColor);

            // this x is used for all row
            var allshowdata_num = 0
            for (let i = 0; i < realldata.length; i++) {
                if (thirdrow_type.includes(realldata[i].interaction_type)) {
                    allshowdata_num = allshowdata_num + 1
                }
            }

            const x = d3.scaleLinear()
                .domain([0, realldata.length])
                .range([0, svgwidth - padding * padding_number]);

            const y = d3.scaleBand()
                .domain(row_array)
                .range([0, svgheight * 1.2])

            const sankey_type = [...firstrow_type, ...secondrow_type, ...thirdrow_type]

            var sankey_position = [padding * padding_number / 2]
            let previous = 0
            for (let i = 1; i <= secondrow_type.length; i++) {
                sankey_position.push(x(previous) + padding * padding_number / (all_modality_type.length + 1) * i)
                previous = secondRowCountData[secondrow_type[i - 1]].detail.length + previous
            }
            previous = 0
            for (let i = 1; i <= thirdrow_type.length; i++) {
                sankey_position.push(padding * padding_number / (all_interaction_type.length + 1) * i + x(previous))
                previous = thirdRowCountData[thirdrow_type[i - 1]].detail.length + previous
            }

            const sankey_x = d3.scaleOrdinal()
                .domain(sankey_type)
                .range(sankey_position)

            const errorbar_height = y.bandwidth() * 0.05;
            const mainbar_height = y.bandwidth() * 0.12;

            // const text_padding = - y.bandwidth() * 0.05;
            // const text_padding = y.bandwidth() * 0.125;
            const text_padding = y.bandwidth() * 0.25;
            // const path_height = y.bandwidth() - errorbar_height - mainbar_height;

            // draw main bar
            const self = this
            for (let key = 0; key < mainbar_type.length; key++) {
                // first row
                let previous_count = 0
                for (let j = 0; j < key; j++) {
                    previous_count = previous_count + firstRowCountData.count[mainbar_type[j]]
                }
                svg.append('rect')
                    .datum({ semantic: mainbar_type[key], data: firstRowCountData.detail })
                    .attr("x", sankey_x("all") + x(previous_count))
                    .attr("y", y(1) + errorbar_height)
                    .attr("width", x(firstRowCountData.count[mainbar_type[key]]))
                    .attr("height", mainbar_height)
                    .attr("fill", color(mainbar_type[key]))
                    .attr("cursor", "pointer")
                    .attr("class", "all" + mainbar_type[key])
                    .classed("conflict" + mainbar_type[key], true)
                    .classed("complement" + mainbar_type[key], true)
                    .classed("nothighlight", true)
                    .classed("U2" + mainbar_type[key], true)
                    .classed("U1" + mainbar_type[key], true)
                    .classed("R" + mainbar_type[key], true)
                    .classed("S" + mainbar_type[key], true)
                    // .classed("undecided" + mainbar_type[key], true)
                    // eslint-disable-next-line
                    .on("mouseover", function (event, d) {
                        if (!this.enableBrush) {
                            d3.selectAll("." + "all" + mainbar_type[key])
                                .classed("nothighlight", false)
                                .attr("opacity", 1)
                            d3.selectAll(".mytext")
                                .classed("nothighlight", false)
                                .attr("opacity", 1)
                            d3.selectAll(".nothighlight")
                                .attr("opacity", 0.3)
                        }
                    })
                    // eslint-disable-next-line
                    .on("mouseout", function (event, d) {
                        if (!this.enableBrush) {
                            d3.selectAll("." + "all" + mainbar_type[key])
                                .classed("nothighlight", true)
                            d3.selectAll(".mytext")
                                .classed("nothighlight", true)
                            d3.selectAll(".nothighlight")
                                .attr("opacity", 1)
                            d3.selectAll(".pathhighlight")
                                .attr("opacity", 0.5)
                        }
                    })
                    .on("click", function (event, d) {
                        self.clickMainbar(d, mainbar_type[key])
                        if (d3.select(this).attr("stroke") === "blue") {
                            self.getData()
                            d3.select(".blued")
                                .attr("stroke", "none")
                                .classed("blued", false)
                            return
                        }
                        d3.select(".blued")
                            .attr("stroke", "none")
                            .classed("blued", false)
                        d3.select(this).attr("stroke", "blue")
                            .attr("stroke-width", 2)
                            .classed("blued", true)
                    })

                // second row
                for (let secondkey = 0; secondkey < secondrow_type.length; secondkey++) {
                    let type = secondrow_type[secondkey]
                    previous_count = 0
                    for (let j = 0; j < key; j++) {
                        previous_count = previous_count + secondRowCountData[type].count[mainbar_type[j]]
                    }
                    svg.append('rect')
                        .datum({ semantic: mainbar_type[key], data: secondRowCountData[type].detail })
                        .attr("x", sankey_x(type) + x(previous_count))
                        .attr("y", y(2) + errorbar_height)
                        .attr("width", x(secondRowCountData[type].count[mainbar_type[key]]))
                        .attr("height", mainbar_height)
                        .attr("fill", color(mainbar_type[key]))
                        .attr("cursor", "pointer")
                        .attr("class", type + mainbar_type[key])
                        .classed("nothighlight", true)
                        .classed("all" + mainbar_type[key], true)
                        .classed("U2" + mainbar_type[key], type == "conflict")
                        .classed("U1" + mainbar_type[key], type == "conflict")
                        .classed("R" + mainbar_type[key], type == "complement")
                        .classed("S" + mainbar_type[key], type == "conflict")
                        // .classed("undecided" + mainbar_type[key], type == "conflict")
                        // eslint-disable-next-line
                        .on("mouseover", function (event, d) {
                            if (!this.enableBrush) {
                                d3.selectAll("." + type + mainbar_type[key])
                                    .classed("nothighlight", false)
                                    .attr("opacity", 1)

                                d3.selectAll(".text" + type)
                                    .attr("opacity", 1)
                                    .attr("font-weight", 600)
                                    .classed("nothighlight", false)
                                // d3.selectAll(".text" + type + "third")
                                //     .attr("opacity", 1)
                                //     .attr("font-weight", 600)
                                //     .classed("nothighlight", false)

                                d3.selectAll(".nothighlight")
                                    .attr("opacity", 0.3)
                            }
                        })
                        // eslint-disable-next-line
                        .on("mouseout", function (event, d) {
                            if (!this.enableBrush) {
                                d3.selectAll("." + type + mainbar_type[key])
                                    .classed("nothighlight", true)
                                d3.selectAll(".text" + type)
                                    .attr("opacity", 0)
                                    .attr("font-weight", 200)
                                    .attr("opacity", 0)
                                //     // .classed("nothighlight", true)
                                // d3.selectAll(".text" + type + "third")
                                //     .attr("font-weight", 200)
                                //     .attr("opacity", 0)
                                    // .classed("nothighlight", true)
                                d3.selectAll(".nothighlight")
                                    .attr("opacity", 1)
                                d3.selectAll(".pathhighlight")
                                    .attr("opacity", 0.5)
                            }
                        })
                        .on("click", function (event, d) {
                            self.clickMainbar(d, mainbar_type[key])
                            if (d3.select(this).attr("stroke") === "blue") {
                                self.getData()
                                d3.select(".blued")
                                    .attr("stroke", "none")
                                    .classed("blued", false)
                                return
                            }
                            d3.select(".blued")
                                .attr("stroke", "none")
                                .classed("blued", false)
                            d3.select(this).attr("stroke", "blue")
                                .attr("stroke-width", 2)
                                .classed("blued", true)
                        })
                }

                // third row
                for (let thirdkey = 0; thirdkey < thirdrow_type.length; thirdkey++) {
                    let type = thirdrow_type[thirdkey]
                    previous_count = 0
                    for (let j = 0; j < key; j++) {
                        previous_count = previous_count + thirdRowCountData[type].count[mainbar_type[j]]
                    }
                    svg.append('rect')
                        .datum({ semantic: mainbar_type[key], data: thirdRowCountData[type].detail })
                        .attr("x", sankey_x(type) + x(previous_count))
                        .attr("y", y(3) + errorbar_height)
                        .attr("width", x(thirdRowCountData[type].count[mainbar_type[key]]))
                        .attr("height", mainbar_height)
                        .attr("fill", color(mainbar_type[key]))
                        .attr("class", type + mainbar_type[key])
                        .attr("cursor", "pointer")
                        .classed(thirdRowCountData[type].detail[0].modality_type + mainbar_type[key], true)
                        .classed("all" + mainbar_type[key], true)
                        .classed("nothighlight", true)
                        // eslint-disable-next-line
                        .on("mouseover", function (event, d) {
                            if (!this.enableBrush) {

                                d3.selectAll("." + type + mainbar_type[key])
                                    .classed("nothighlight", false)
                                    .attr("opacity", 1)
                                d3.selectAll("." + thirdRowCountData[type].detail[0].modality_type + mainbar_type[key] + "path")
                                    .classed("nothighlight", false)
                                    .attr("opacity", 1)

                                d3.selectAll(".text" + type)
                                    .attr("opacity", 1)
                                    .attr("font-weight", 600)
                                    .classed("nothighlight", false)
                                d3.selectAll(".text" + thirdRowCountData[type].detail[0].modality_type)
                                    .attr("opacity", 1)
                                    .attr("font-weight", 600)
                                    .classed("nothighlight", false)

                                d3.selectAll(".nothighlight")
                                    .attr("opacity", 0.3)
                            }
                        })
                        // eslint-disable-next-line
                        .on("mouseout", function (event, d) {
                            if (!this.enableBrush) {
                                d3.selectAll("." + type + mainbar_type[key])
                                    .classed("nothighlight", true)
                                d3.selectAll(".text" + type)
                                    .attr("opacity", 0)
                                    .attr("font-weight", 200)
                                    .attr("opacity", 0)
                                    // .classed("nothighlight", true)
                                d3.selectAll(".text" + thirdRowCountData[type].detail[0].modality_type)
                                    .attr("opacity", 0)
                                    .attr("font-weight", 200)
                                    .attr("opacity", 0)
                                    // .classed("nothighlight", true)
                                d3.selectAll("." + thirdRowCountData[type].detail[0].modality_type + mainbar_type[key] + "path")
                                    .attr("opacity", 0.5)
                                d3.selectAll(".nothighlight")
                                    .attr("opacity", 1)
                                d3.selectAll(".pathhighlight")
                                    .attr("opacity", 0.5)
                            }
                        })
                        .on("click", function (event, d) {
                            self.clickMainbar(d, mainbar_type[key])
                            if (d3.select(this).attr("stroke") === "blue") {
                                self.getData()
                                d3.select(".blued")
                                    .attr("stroke", "none")
                                    .classed("blued", false)
                                return
                            }
                            d3.select(".blued")
                                .attr("stroke", "none")
                                .classed("blued", false)
                            d3.select(this).attr("stroke", "blue")
                                .attr("stroke-width", 2)
                                .classed("blued", true)
                        })
                }
            }

            // draw error bar
            // first row error bar
            svg.selectAll("errorbar")
                .data(firstRowCountData.detail)
                .enter()
                .append("rect")
                .attr("x", function (d, i) { return sankey_x("all") + x(i) })
                .attr("y", y(1))
                .attr("width", x(1) - x(0))
                .attr("height", errorbar_height)
                .style("fill", function (d) {
                    if (d.gt == -1 && d.prediction.overall == "negative") {
                        return errorcolor(true)
                    } else if (d.gt == 1 && d.prediction.overall == "positive") {
                        return errorcolor(true)
                    } else if (d.gt == 0 && d.prediction.overall == "neutral") {
                        return errorcolor(true)
                    } else {
                        return errorcolor(false)
                    }
                })
                .attr("class", function (d) { return "all" + d.prediction.overall })
                .each(function (d) {
                    d3.select(this).classed("select", true)
                    d3.select(this).classed("allerror" + d.videoname, true);
                    d3.select(this).classed(d.modality_type + d.prediction.overall, true);
                    d3.select(this).classed(d.videoname, true);
                    d3.select(this).classed("all" + d.prediction.overall, true);
                    d3.select(this).classed(d.interaction_type + d.prediction.overall, true)
                })
                .classed("nothighlight", true)

            // second row error bar
            for (let secondkey = 0; secondkey < secondrow_type.length; secondkey++) {
                let type = secondrow_type[secondkey]
                svg.selectAll("errorbar")
                    .data(secondRowCountData[type].detail)
                    .enter()
                    .append("rect")
                    .attr("x", function (d, i) { return x(i) + sankey_x(type) })
                    .attr("y", y(2))
                    .attr("width", x(1) - x(0))
                    .attr("height", errorbar_height)
                    .style("fill", function (d) {
                        if (d.gt == -1 && d.prediction.overall == "negative") {
                            return errorcolor(true)
                        } else if (d.gt == 1 && d.prediction.overall == "positive") {
                            return errorcolor(true)
                        } else if (d.gt == 0 && d.prediction.overall == "neutral") {
                            return errorcolor(true)
                        } else {
                            return errorcolor(false)
                        }
                    })
                    .attr("class", function (d) { return type + d.prediction.overall })
                    .each(function (d) {
                        d3.select(this).classed("select", true)
                        d3.select(this).classed(type + "error" + d.videoname, true);
                        d3.select(this).classed(d.modality_type + d.prediction.overall, true);
                        d3.select(this).classed(d.videoname, true);
                        d3.select(this).classed("all" + d.prediction.overall, true);
                        d3.select(this).classed(d.interaction_type + d.prediction.overall, true)
                    })
                    .classed("nothighlight", true)
            }

            // third row error bar
            for (let thirdkey = 0; thirdkey < thirdrow_type.length; thirdkey++) {
                let type = thirdrow_type[thirdkey]
                svg.selectAll("errorbar")
                    .data(thirdRowCountData[type].detail)
                    .enter()
                    .append("rect")
                    .attr("x", function (d, i) { return x(i) + sankey_x(type) })
                    .attr("y", y(3))
                    .attr("width", x(1) - x(0))
                    .attr("height", errorbar_height)
                    .style("fill", function (d) {
                        if (d.gt == -1 && d.prediction.overall == "negative") {
                            return errorcolor(true)
                        } else if (d.gt == 1 && d.prediction.overall == "positive") {
                            return errorcolor(true)
                        } else if (d.gt == 0 && d.prediction.overall == "neutral") {
                            return errorcolor(true)
                        } else {
                            return errorcolor(false)
                        }
                    })
                    .attr("class", function (d) { return type + d.prediction.overall })
                    .each(function (d) {
                        d3.select(this).classed("select", true)
                        d3.select(this).classed(type + "error" + d.videoname, true);
                        d3.select(this).classed(d.modality_type + d.prediction.overall, true);
                        d3.select(this).classed(d.videoname, true);
                        d3.select(this).classed("all" + d.prediction.overall, true);
                        d3.select(this).classed(d.interaction_type + d.prediction.overall, true)
                    })
                    .classed("nothighlight", true)
            }

            // draw path
            var pathdata = []
            let all_previous = 0
            for (let thirdkey = 0; thirdkey < thirdrow_type.length; thirdkey++) {
                if (thirdkey == 2){
                    all_previous = 0
                }
                let previous = 0
                // let all_previous = 0
                for (let key = 0; key < mainbar_type.length; key++) {
                    let type = thirdrow_type[thirdkey]
                    if (thirdRowCountData[type].count[mainbar_type[key]] !== 0) {
                        let startpoint = {
                            x: x(thirdRowCountData[type].count[mainbar_type[key]] / 2) + sankey_x(type) + x(previous),
                            y: y(3),
                            w: x(thirdRowCountData[type].count[mainbar_type[key]]),
                            gt: mainbar_type[key],
                            class: type + mainbar_type[key],
                            class2: thirdRowCountData[type].detail[0].modality_type + mainbar_type[key],
                            class3: "all" + mainbar_type[key],
                            class4: "all" + mainbar_type[key],
                        }
                        let compensation = 0
                        // for (let j = 1; j < thirdkey; j++) {
                        //     compensation = compensation + thirdRowCountData[thirdrow_type[j]].count[mainbar_type[key]]
                        // }
                        if (thirdkey === 1 || thirdkey === 3){
                            compensation = compensation - secondRowCountData[thirdRowCountData[type].detail[0].modality_type].count["negative"] - thirdRowCountData[type].count["all"]
                            console.log("compensation showing", compensation)
                        }
                        let endpoint = {
                            x: x(thirdRowCountData[type].count[mainbar_type[key]] / 2) + sankey_x(thirdRowCountData[type].detail[0].modality_type) + x(all_previous) + x(compensation),
                            y: y(2) + errorbar_height + mainbar_height,
                            w: x(thirdRowCountData[type].count[mainbar_type[key]]),
                            gt: mainbar_type[key],
                            class: type + mainbar_type[key],
                            class2: thirdRowCountData[type].detail[0].modality_type + mainbar_type[key],
                            class3: "all" + mainbar_type[key],
                        }
                        previous = previous + thirdRowCountData[type].count[mainbar_type[key]]
                        all_previous = all_previous + secondRowCountData[thirdRowCountData[type].detail[0].modality_type].count[mainbar_type[key]]
                        pathdata.push([startpoint, endpoint])
                    }
                }
            }

            for (let secondkey = 0; secondkey < secondrow_type.length; secondkey++) {
                let previous = 0
                let all_previous = 0
                for (let key = 0; key < mainbar_type.length; key++) {
                    let type = secondrow_type[secondkey]
                    if (secondRowCountData[type].count[mainbar_type[key]] !== 0) {
                        let startpoint = {
                            x: x(secondRowCountData[type].count[mainbar_type[key]] / 2) + sankey_x(type) + x(previous),
                            y: y(2),
                            w: x(secondRowCountData[type].count[mainbar_type[key]]),
                            gt: mainbar_type[key],
                            class: type + mainbar_type[key],
                            class2: "all" + mainbar_type[key],
                            class3: type + mainbar_type[key] + "path",
                        }
                        let compensation = 0
                        for (let j = 0; j < secondkey; j++) {
                            compensation = compensation + secondRowCountData[secondrow_type[j]].count[mainbar_type[key]]
                        }
                        let endpoint = {
                            x: x(secondRowCountData[type].count[mainbar_type[key]] / 2) + sankey_x("all") + x(all_previous) + x(compensation),
                            y: y(1) + errorbar_height + mainbar_height,
                            w: x(secondRowCountData[type].count[mainbar_type[key]]),
                            gt: mainbar_type[key],
                            class: type + mainbar_type[key],
                            class2: "all" + mainbar_type[key],
                            class3: type + mainbar_type[key] + "path",
                        }
                        previous = previous + secondRowCountData[type].count[mainbar_type[key]]
                        all_previous = all_previous + firstRowCountData.count[mainbar_type[key]]
                        pathdata.push([startpoint, endpoint])
                    }
                }
            }

            var lineGenerator = d3.line()
                .x(d => d.x)
                .y(d => d.y)
                .curve(d3.curveBumpY);

            pathdata.forEach(pathdata => {
                svg.append("path")
                    .datum(pathdata)
                    .attr("fill", "none")
                    // .attr("stroke", color(pathdata[0].gt))
                    .attr("stroke", "#D2D2D2")
                    .attr("stroke-width", pathdata[0].w)
                    .attr("opacity", 0.5)
                    .attr("d", lineGenerator)
                    .attr("class", pathdata[0].class)
                    .classed("pathhighlight", true)
                    .classed(pathdata[0].class2, true)
                    .classed(pathdata[0].class3, true)
            });

            // draw other modality
            const modalitybar_padding = y(3) + errorbar_height + mainbar_height + 0.02 * svgheight;
            const modalitybar_height = mainbar_height;

            const y_modality = d3.scaleLinear()
                .domain([0, 1])
                .range([modalitybar_padding, modalitybar_padding + errorbar_height + mainbar_height])
            for (let thirdkey = 0; thirdkey < thirdrow_type.length; thirdkey++) {
                let type = thirdrow_type[thirdkey]
                svg.selectAll("modalitybar")
                    .data(thirdRowCountData[type].detail)
                    .enter()
                    .append("rect")
                    .attr("x", function (d, i) { return x(i) + sankey_x(type) })
                    .attr("y", y_modality(0))
                    .attr("width", x(1) - x(0))
                    .attr("height", modalitybar_height)
                    .attr("fill", function (d) { return color(d.prediction.language) })
                    .attr("class", function (d) { return type + d.prediction.overall })
                    .each(function (d) {
                        d3.select(this).classed(d.modality_type + d.prediction.overall, true);
                        d3.select(this).classed(d.videoname, true);
                        d3.select(this).classed("all" + d.prediction.overall, true);
                        d3.select(this).classed(d.interaction_type + d.prediction.overall, true);
                        d3.select(this).classed("select", true);
                    })
                    .classed("nothighlight", true)

                svg.selectAll("modalitybar")
                    .data(thirdRowCountData[type].detail)
                    .enter()
                    .append("rect")
                    .attr("x", function (d, i) { return x(i) + sankey_x(type) })
                    .attr("y", y_modality(1))
                    .attr("width", x(1) - x(0))
                    .attr("height", modalitybar_height)
                    .attr("fill", function (d) { return color(d.prediction.visual) })
                    .attr("class", function (d) { return type + d.prediction.overall })
                    .each(function (d) {
                        d3.select(this).classed(d.modality_type + d.prediction.overall, true);
                        d3.select(this).classed(d.videoname, true);
                        d3.select(this).classed("all" + d.prediction.overall, true);
                        d3.select(this).classed(d.interaction_type + d.prediction.overall, true);
                        d3.select(this).classed("select", true);
                    })
                    .classed("nothighlight", true)
            }

            // legend

            // svg.append('text')
            //     .attr("x", sankey_x("all") + x(firstRowCountData.count.all / 2))
            //     .attr("y", y(1) + text_padding)
            //     .text("all")
            for (let secondkey = 0; secondkey < secondrow_type.length; secondkey++) {
                let type = secondrow_type[secondkey]
                // hovering text
                // svg.append('text')
                //     .attr("x", sankey_x(type) + x(secondRowCountData[type].count.all / 2))
                //     .attr("y", y(2) + text_padding)
                //     .attr("text-anchor", "middle")
                //     .attr("pointer-events", "none")
                //     .attr("alignment-baseline", "middle")
                //     .text(type)
                //     .attr("font-weight", 200)
                //     .attr("opacity", 0)
                //     .attr("class", "text" + type)
                //     .classed("mytext", true)
                    // .classed("nothighlight", true)
                
                //static text
                svg.append('text')
                    .attr("x", sankey_x(type) + x(secondRowCountData[type].count.all / 2))
                    .attr("y", y(2) + text_padding)
                    .attr("alignment-baseline", "middle")
                    .text(type)
                    .attr("font-weight", 200)
                    // .attr("opacity", 0)
                    // .attr("class", "text" + type)
                    // .classed("mytext", true)
            }
            for (let thirdkey = 0; thirdkey < thirdrow_type.length; thirdkey++) {
                let type = thirdrow_type[thirdkey]
                // hovering text
                // svg.append('text')
                //     .attr("x", sankey_x(type) + x(thirdRowCountData[type].count.all / 2))
                //     .attr("y", y(3) + text_padding)
                //     .attr("text-anchor", "middle")
                //     .attr("alignment-baseline", "middle")
                //     .attr("pointer-events", "none")
                //     .attr("font-weight", 200)
                //     .attr("opacity", 0)
                //     .attr("class", "text" + type)
                //     // .classed("nothighlight", true)
                //     .classed("mytext", true)
                //     .classed("text" + thirdRowCountData[type].detail[0].modality_type + "third", true)
                //     .text(function () {
                //         if (type == "undecided") {
                //             return "TBD";
                //         } else {
                //             return type
                //         }
                //     })
                //static text
                svg.append('text')
                    .attr("x", sankey_x(type) + x(thirdRowCountData[type].count.all / 2))
                    .attr("y", y(3) + text_padding + mainbar_height*2 + errorbar_height*1.5)
                    .attr("text-anchor", "middle")
                    .attr("alignment-baseline", "middle")
                    .attr("pointer-events", "none")
                    .attr("font-weight", 200)
                    // .attr("opacity", 0)
                    // .attr("class", "text" + type)
                    // // .classed("nothighlight", true)
                    // .classed("mytext", true)
                    // .classed("text" + thirdRowCountData[type].detail[0].modality_type + "third", true)
                    .text(function () {
                        if (type == "undecided") {
                            return "TBD";
                        } else {
                            if (namechange(type) == "Distinct") {
                                return "";
                            }
                            return namechange(type)
                        }
                    })
            }
            legend_svg.append('text')
                .text("A:")
                .attr("x", margin.left)
                .attr("y", y_modality(0) - modalitybar_height)
            legend_svg.append('text')
                .text("L:")
                .attr("x", margin.left)
                .attr("y", y_modality(0) + modalitybar_height / 2)
            legend_svg.append('text')
                .text("V:")
                .attr("x", margin.left)
                .attr("y", y_modality(1) + modalitybar_height / 2)

        },
        clickMainbar(d, type) {
            var data = d.data
            var filteredData = data.filter(function (d) {
                return d.prediction.overall === type;
            });
            var video_list = []
            for (let i in filteredData) {
                video_list.push(filteredData[i].videoname)
            }
            if (video_list.length < 5) {
                alert('This type has less than 5 instances. Cannot get concepts!')
            }
            this.fetchUpdateData(video_list)
            this.$emit('instancedata', video_list);
        },
        fetchData() {
            this.myTable = []
            for (let i = 0; i < this.allData.length; i++) {
                let oneinstance = this.allData[i].itemset_list
                let instancedata = {}
                let one_language = []
                let one_visual = []
                for (let j = 0; j < oneinstance.length; j++) {
                    if (oneinstance[j][0] == "l") {
                        one_language.push({
                            key: "instance" + i.toString() + "language" + j.toString(),
                            concept: oneinstance[j],
                            concept_id: oneinstance[j][oneinstance[j].length - 1],
                            represent: this.allData[i].itemsets[oneinstance[j]].represent_cue_text
                        })
                    } else {
                        var now_j = j - one_language.length
                        one_visual.push({
                            key: "instance" + i.toString() + "visual" + now_j.toString(),
                            concept: oneinstance[j],
                            concept_id: oneinstance[j][oneinstance[j].length - 1],
                            represent: this.allData[i].itemsets[oneinstance[j]].represent_cue_text
                        })
                    }
                }
                instancedata.language = one_language
                instancedata.visual = one_visual
                instancedata.alldata = this.allData[i]
                instancedata.instance_number = this.allData[i].video_instances.length
                instancedata.initial_index = i

                var error_number = 0

                for (let j = 0; j < this.allData[i].video_instances.length; j++) {
                    const video_id = this.allData[i].video_instances[j]
                    if (this.allInstanceData[video_id].gt === 0 && this.allInstanceData[video_id].prediction.overall.toLowerCase() === "neutral") {
                        continue
                    } else if (this.allInstanceData[video_id].gt === 1 && this.allInstanceData[video_id].prediction.overall.toLowerCase() === "positive") {
                        continue
                    } else if (this.allInstanceData[video_id].gt === -1 && this.allInstanceData[video_id].prediction.overall.toLowerCase() === "negative") {
                        continue
                    } else {
                        error_number = error_number + 1
                    }
                }

                instancedata.error_number = error_number

                this.myTable.push(instancedata)
            }
        },
        toggleBrush() {
            this.enableBrush = !this.enableBrush;
            if (this.enableBrush) {
                this.createBrush();
            } else {
                this.disableBrush();
                this.getData()
            }
        },
        disableBrush() {
            const svg = d3.select("#pathsvg");
            svg.select(".brush").remove(); // 移除brush功能

            svg.selectAll(".select")
                // .attr("stroke", "none")
                // .attr("stroke-width", "0px")

                .attr("opacity", 1)
                .classed("selected", false)
        },
        createBrush() {
            const svgElement = document.getElementById("pathsvg");

            const margin = { top: 20, right: 10, bottom: 2, left: 2 };
            const svgwidth = svgElement.clientWidth;
            const svgheight = svgElement.clientHeight;
            const legend_width = 0.025 * svgwidth

            const brush = d3.brush()
                .extent([[0, 0], [svgwidth, svgheight]])
                .on("brush", brushed)
                .on("end", brushended);
            const svg = d3.select("#pathsvg")

            svg.append("g")
                .attr("class", "brush")
                .call(brush);

            function brushed(event) {
                const selection = event.selection;
                if (selection === null) return;

                const [[x0, y0], [x1, y1]] = selection;

                svg.selectAll(".select")
                    // .attr("stroke", "none")
                    // .attr("stroke-width", "0px")
                    .attr("opacity", 0.3)
                    .classed("selected", false)

                svg.selectAll(".select")
                    .classed("selected", function () {
                        const rectX = parseFloat(d3.select(this).attr("x"));
                        const rectY = parseFloat(d3.select(this).attr("y"));
                        const rectWidth = parseFloat(d3.select(this).attr("width"));
                        const rectHeight = parseFloat(d3.select(this).attr("height"));
                        const rectX2 = rectX + rectWidth;
                        const rectY2 = rectY + rectHeight;

                        const intersects = (rectX < x1 - legend_width - margin.left && rectX2 > x0 - legend_width - margin.left && rectY < y1 - margin.top && rectY2 > y0 - margin.top);
                        return intersects;
                    })
                d3.selectAll(".selected")
                    .attr("opacity", 1)
                // .attr("stroke", "black")
                // .attr("stroke-width", "0.5px");
            }
            const self = this
            function brushended() {
                // if (!event.selection) return;
                const selectedItems = svg.selectAll("rect.selected").data();
                // 这里可以处理选中的数据
                // console.log(selectedItems);
                var video_list = []

                for (let i = 0; i < selectedItems.length; i++) {
                    if (!video_list.includes(selectedItems[i].videoname)){
                        video_list.push(selectedItems[i].videoname)
                    }
                }

                if (video_list.length < 5) {
                    alert('Please brush 5 or more instances!')
                    return
                }

                self.fetchUpdateData(video_list)
                console.log("brush", video_list);
                self.$emit('instancedata', video_list);
                // 如果需要在结束时取消选择
                // svg.select(".brush").call(brush.move, null);
            }
        },
        async fetchUpdateData(video) {
            var results = await requesthelp.axiosPost("/compute_frequent_sets", {
                video_list: video
            })
            // console.log("updated", results)
            this.allData = results["frequent_itemsets"]
            this.fetchData()
            this.renderAll()
        },
        renderRatio() {
            var all_count = 0;
            for (let i = 0; i < this.allData.length; i++) {
                if (this.allData[i].video_instances.length > all_count) {
                    all_count = this.allData[i].video_instances.length
                }
            }

            for (let i = 0; i < this.myTable.length; i++) {
                var count = 0
                for (let j = 0; j < this.myTable[i].language.length; j++) {
                    let svg = 'instance' + i.toString() + "language" + j.toString()
                    this.drawRatio(svg, i, j, "language", all_count)
                    count = count + 1
                }
                for (let j = 0; j < this.myTable[i].visual.length; j++) {
                    let svg = 'instance' + i.toString() + "visual" + j.toString()
                    this.drawRatio(svg, i, (j + count), "visual", all_count)
                }
            }
        },
        drawRatio(refKey, instance_id, concept_id, type, all_count) {
            this.$nextTick(() => {
                const concept = this.allData[instance_id]["itemset_list"][concept_id]
                const conceptdata = this.allData[instance_id]["itemsets"][concept]
                const videodata = this.allData[instance_id]["video_instances"]

                var positive_count = 0
                var neutral_count = 0
                var negative_count = 0
                if (type == "language") {
                    for (let i = 0; i < videodata.length; i++) {
                        for (let j = 0; j < this.allInstanceData[videodata[i]].reasoning.language_cues.length; j++) {
                            if (conceptdata["cue_list"].includes(this.allInstanceData[videodata[i]].reasoning.language_cues[j][0])) {
                                // console.log(j)
                                if (this.allInstanceData[videodata[i]].reasoning.language_cues[j][1] == "positive") {
                                    positive_count = positive_count + 1
                                } else if (this.allInstanceData[videodata[i]].reasoning.language_cues[j][1] == "neutral") {
                                    neutral_count = neutral_count + 1
                                } else {
                                    negative_count = negative_count + 1
                                }
                            }
                        }
                    }
                } else {
                    for (let i = 0; i < videodata.length; i++) {
                        for (let j = 0; j < this.allInstanceData[videodata[i]].reasoning.visual_cues.length; j++) {
                            if (conceptdata["cue_list"].includes(this.allInstanceData[videodata[i]].reasoning.visual_cues[j][0])) {
                                // console.log(j)
                                if (this.allInstanceData[videodata[i]].reasoning.visual_cues[j][1] == "positive") {
                                    positive_count = positive_count + 1
                                } else if (this.allInstanceData[videodata[i]].reasoning.visual_cues[j][1] == "neutral") {
                                    neutral_count = neutral_count + 1
                                } else {
                                    negative_count = negative_count + 1
                                }
                            }
                        }
                    }
                }
                // var all_count = positive_count + neutral_count + negative_count

                const el = this.$refs[refKey];
                // const svgElement = el.length ? el[0] : el; // 获取第一个元素
                const width = el[0].clientWidth;
                // const height = el[0].clientHeight;

                var margin = { top: 2, right: 2, bottom: 2, left: 2 };
                var svgwidth = width - margin.right - margin.left;
                // var svgheight = width - margin.top - margin.bottom;

                var svg = d3.select('#' + refKey)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


                const color = d3.scaleOrdinal()
                    .domain(["positive", "neutral", "negative"])
                    .range(this.colorarray);

                if (all_count != 0) {
                    var x = d3.scaleLinear()
                        .domain([0, all_count])
                        .range([0, svgwidth]);

                    svg.append("rect")
                        .attr("x", x(0))
                        .attr("y", 0)
                        .attr("width", x(positive_count))
                        .attr("height", 15)
                        .attr("fill", color("positive"))

                    svg.append("rect")
                        .attr("x", x(positive_count))
                        .attr("y", 0)
                        .attr("width", x(neutral_count))
                        .attr("height", 15)
                        .attr("fill", color("neutral"))

                    svg.append("rect")
                        .attr("x", x(positive_count + neutral_count))
                        .attr("y", 0)
                        .attr("width", x(negative_count))
                        .attr("height", 15)
                        .attr("fill", color("negative"))
                }
            });
        },
        handleCellclick(row) {
            // console.log('Cell clicked:', row, column, cell);
            console.log("cell", row.alldata.video_instances);
            this.$emit('instancedata', row.alldata.video_instances);
        },
        // eslint-disable-next-line
        getCellClassName({ rowIndex, columnIndex }) {
            if (columnIndex === 1 || columnIndex === 2) {
                return 'clickable-cell';
            }
            return '';
        },
        handleCurrentChangeLan(newPage, initrow) {
            console.log(newPage);
            let svg = 'lansvg' + initrow.toString()
            this.drawWordcloud(svg, initrow, newPage - 1, "language_cues")
        },
        handleCurrentChangeVis(newPage, initrow) {
            console.log(newPage);
            let svg = 'vissvg' + initrow.toString()
            this.drawWordcloud(svg, initrow, newPage - 1, "visual_cues")
        },
        onTableRowExpand(row) {
            const index = row.initial_index

            // this.drawWordcloud('svg' + index.toString())
            // const count = 0
            // for (let j = 0; j < this.myTable[index].language.length; j++) {
            let svg = 'lansvg' + index.toString()
            this.drawWordcloudNext(svg, index, 0, "language_cues")
            // count = count + 1
            // }
            // for (let j = 0; j < this.myTable[index].visual.length; j++) {
            let svg2 = 'vissvg' + index.toString()
            this.drawWordcloudNext(svg2, index, 0, "visual_cues")
            // }
        },
        drawWordcloudNext(refKey, instance_id, concept_id, type) {
            console.log(refKey)
            this.$nextTick(() => {
                this.drawWordcloud(refKey, instance_id, concept_id, type)
            });
        },
        drawWordcloud(refKey, instance_id, concept_id, type) {
            const el = document.getElementById(refKey)
            console.log(refKey)
            if (!el) {
                return
            }
            const width = el.clientWidth;
            const height = el.clientHeight;
            var margin = { top: 2, right: 2, bottom: 2, left: 0 };
            var svgwidth = width - margin.right - margin.left
            var svgheight = height - margin.top - margin.bottom;
            d3.select('#' + refKey).selectAll("*").remove();
            var svg = d3.select('#' + refKey)
                // .append("g")
                // .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
                .append("g") // 创建一个 g 元素作为容器
                .attr("transform", "translate(" + svgwidth / 2 + "," + svgheight / 2 + ")");

            if (type === "visual_cues"){
                for (let i = 0; i < this.allData[instance_id]["itemset_list"].length; i++) {
                    if (this.allData[instance_id]["itemset_list"][i][0] === "l") {
                        continue
                    } else {
                        concept_id = i
                        break
                    }
                }
            }

            const concept = this.allData[instance_id]["itemset_list"][concept_id]
            const conceptdata = this.allData[instance_id]["itemsets"][concept]
            const videodata = this.allData[instance_id]["video_instances"]
            const cuelist = conceptdata["cue_list"]
            console.log(concept, conceptdata, videodata, this.allInstanceData)

            var wordclouddata = {}
            for (let i = 0; i < cuelist.length; i++) {
                wordclouddata[cuelist[i]] = {
                    count: 0,
                    positive: 0,
                    negative: 0,
                    neutral: 0
                }
            }

            for (let i = 0; i < videodata.length; i++) {
                const key = videodata[i]
                for (let j = 0; j < this.allInstanceData[key].reasoning[type].length; j++) {
                    let thiscue = this.allInstanceData[key].reasoning[type][j][0]
                    let thissem = this.allInstanceData[key].reasoning[type][j][1]
                    if (cuelist.includes(thiscue)) {
                        wordclouddata[thiscue].count = wordclouddata[thiscue].count + 1
                        wordclouddata[thiscue][thissem] = wordclouddata[thiscue][thissem] + 1
                    }
                }
            }
            console.log(wordclouddata)

            let wordsArray = Object.keys(wordclouddata).map(key => {
                return { text: key, size: wordclouddata[key].count }; // or any other metric you choose
            });
            const self = this
            function interpolateColor(value) {
                // 定义基础颜色
                // const red = d3.rgb(255, 0, 0); // 红色
                const red = self.colorarrayrgb[0]
                // const yellow = d3.rgb(255, 255, 0); // 黄色
                const yellow = self.colorarrayrgb[1]
                // const blue = d3.rgb(0, 0, 255); // 蓝色
                const blue = self.colorarrayrgb[2]
                
                // 计算总数
                const total = value.positive + value.neutral + value.negative;

                // 计算比例
                const redRatio = (value.positive / total);
                const yellowRatio = (value.neutral / total);
                const blueRatio = (value.negative / total);

                if (value.positive === value.neutral && value.negative === value.neutral){
                    var mixedColor = d3.rgb(201, 201, 201)
                    return mixedColor.toString();
                }

                // 计算颜色
                mixedColor = d3.rgb(
                    red.r * redRatio + yellow.r * yellowRatio + blue.r * blueRatio,
                    red.g * redRatio + yellow.g * yellowRatio + blue.g * blueRatio,
                    red.b * redRatio + yellow.b * yellowRatio + blue.b * blueRatio
                );

                return mixedColor.toString();
            }


            // Set up the dimensions for your SVG.
            // let svgheight = 500; // Set the height of the SVG element (example value)
            // let svg = d3.select('#' + refKey)
            //     .append("svg")
            //     .attr("width", svgwidth)
            //     .attr("height", svgheight)
            //     .append("g")
            //     .attr("transform", "translate(" + svgwidth / 2 + "," + svgheight / 2 + ")");

            // Set up the d3-cloud layout.
            let layout = cloud()
                .size([svgwidth, svgheight])
                .words(wordsArray)
                .padding(1)
                // .rotate(() => ~~(Math.random() * 2) * 90)
                .rotate(0)
                .font("Impact")
                .fontSize(d => (d.size + 14))
                .on("end", draw);

            // Start the word cloud layout.
            layout.start();

            // Define the draw function.
            function draw(words) {
                svg
                    .selectAll("text")
                    .data(words)
                    .enter().append("text")
                    .style("font-size", d => d.size + "px")
                    .style("font-family", "sans-serif")
                    .style("fill", d => interpolateColor(wordclouddata[d.text])) 
                    .attr("text-anchor", "middle")
                    .attr("transform", d => "translate(" + [d.x, d.y] + ")")
                    // .attr("transform", d => "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")")
                    .text(d => d.text);
            }
        }
    }

};
</script>

<style>
.reasoningpathView {
    /* background-color: lightgray; */
    width: 100%;
    height: 100%;
    display: flex;
    padding-top: 5px;
}

.reasoningpathContainer {
    /* background-color: lightgray; */
    width: 100%;
    height: 47vh;
}

.pathleft {
    height: 100%;
    width: 40%;
}

.pathright {
    height: 100%;
    width: 60%;
}

.righttable {
    height: 99% !important;
    width: 100% !important;
    overflow-y: auto !important;
    /* cursor: pointer; */
}

.righttable .clickable-cell:hover {
    cursor: pointer;
    color: #409EFF;
    /* Element UI 默认蓝色，可以根据需要自定义 */
}

.righttable .el-table .border-right .cell {
    border-right: 1px solid #ebeef5 !important;
}

.righttable .el-table .border-left .cell {
    border-left: 1px solid #ebeef5 !important;
}
</style>
