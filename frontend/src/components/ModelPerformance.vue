<template>
    <div class="modelperformanceContainer viewbottom view">
        <div class="performance">
            <div class="selectTitle">Overall Model Performance</div>
            <div class="performanceView">
                <el-table ref="evaluationTable" :data="evaluationTable" style="width: 100%" height="100%"
                    @expand-change="onTableRowExpand">
                    <el-table-column type="expand" width="35px" fixed>
                        <template v-slot="scope">
                            <svg :ref="`svg${scope.row.versionId}`" :id="`svg${scope.row.versionId}`"
                                style="width:80%; height: 120px">
                            </svg>
                        </template>
                    </el-table-column>
                    <el-table-column label="Version" fixed>
                        <template v-slot="scope">
                            {{ scope.row.version }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="acc" label="Acc" width="90px" sortable fixed>
                        <template v-slot="scope">
                            {{ scope.row.acc }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="f1" label="F1" width="90px" sortable fixed>
                        <template v-slot="scope">
                            {{ scope.row.f1 }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="precision" label="Precision" sortable width="110px" fixed>
                        <template v-slot="scope">
                            {{ scope.row.precision }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="recall" label="Recall" width="110px" sortable fixed>
                        <template v-slot="scope">
                            {{ scope.row.recall }}
                        </template>
                    </el-table-column>
                </el-table>
            </div>
        </div>
    </div>
</template>

<script setup>
import {
    ElTable,
    ElTableColumn
} from 'element-plus'
</script>

<script>
// import requesthelp from "../service/request.js";
import * as d3 from 'd3';

export default {
    props: ['allInstancedata', 'newdata'],
    data() {
        return {
            evaluationResults: {},
            evaluationTable: [],
            versionId: 1,
            // true, false
            // color: ["#8BAD83","#CB7B78"]
            color: ["#52694D","#BCCBB9"]
        };
    },
    mounted() {

    },
    watch: {
        allInstancedata() {
            this.evaluationResults = this.allInstancedata["initial_model_performance"]
            console.log("initial performance", this.evaluationResults)
            this.evaluationTable.push({
                version: "Version 1",
                versionId: 1,
                acc: this.evaluationResults.acc.toFixed(2),
                f1: this.evaluationResults.f1.toFixed(2),
                precision: this.evaluationResults.precision.toFixed(2),
                recall: this.evaluationResults.recall.toFixed(2),
                confusion_matrix: this.evaluationResults.confusion_matrix
            })
        },
        newdata(v) {
            console.log("new data", v)
            const performance = v["model_performance"]
            this.versionId = this.versionId + 1
            this.evaluationTable.push({
                version: "Version " + this.versionId.toString(),
                versionId: this.versionId,
                acc: performance.acc.toFixed(2),
                f1: performance.f1.toFixed(2),
                precision: performance.precision.toFixed(2),
                recall: performance.recall.toFixed(2),
                confusion_matrix: performance.confusion_matrix
            })
        }
    },
    methods: {
        onTableRowExpand(row) {
            const index = row.versionId
            this.drawConfusion(index)
        },
        drawConfusion(index) {
            this.$nextTick(() => {
                const data = this.evaluationTable[index - 1]["confusion_matrix"]
                const refKey = 'svg' + index.toString()
                const el = document.getElementById(refKey)
                console.log("drawConfusion", data, refKey, el)
                if (!el) {
                    return
                }
                console.log("drawConfusion", data, refKey, el)
                const width = el.clientWidth;
                const height = el.clientHeight;
                var margin = { top: 2, right: 0, bottom: 2, left: 35 };
                var svgwidth = width - margin.right - margin.left
                var svgheight = height - margin.top - margin.bottom;
                d3.select('#' + refKey).selectAll("*").remove();

                var maxcount = 0
                for (let i = 0; i < data.length; i++) {
                    for (let j = 0; j < data[i].length; j++) {
                        if (maxcount <= data[i][j])
                        maxcount = data[i][j]
                    }
                }

                const svg = d3.select('#' + refKey)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

                const legendpadding = 30
                const legendpaddingwidth = 20
                    
                const x = d3.scaleBand()
                    .range([legendpadding, svgwidth])
                    .domain([0, 1, 2])
                    .padding(0.05);
                const y = d3.scaleBand()
                    .range([legendpaddingwidth, svgheight])
                    .domain([0, 1, 2])
                    .padding(0.05);
                
                const mycolor = d3.scaleOrdinal()
                    .domain([true, false])
                    .range(this.color)
                const linearcolor = d3.scaleLinear()
                    .domain([0,maxcount])
                    // .range([0.3,1])
                    .range([0.1,1])

                for (let i = 0; i < data.length; i++) {
                    for (let j = 0; j < data[i].length; j++) {
                        svg.datum(data[i][j])
                            .append("rect")
                            .attr("x", function (d) { return x(j) })
                            .attr("y", function (d) { return y(i) })
                            .attr("width", x.bandwidth())
                            .attr("height", y.bandwidth())
                            // .style("fill", function (d) { return mycolor(i==j) })
                            .style("fill", "#52694D")
                            .style("stroke-width", 4)
                            .style("stroke", "none")
                            .style("opacity", function (d) { return linearcolor(d) })
                        
                        svg.datum(data[i][j])
                            .append("text")
                            .attr("x", function (d) { return (x(j)+x.bandwidth()/2) })
                            .attr("y", function (d) { return (y(i)+y.bandwidth()/2) })
                            .text( function (d) { return d })
                            .attr("text-anchor", "middle")
                            .attr("alignment-baseline", "middle")
                            .attr("pointer-events", "none")
                            .attr("font-weight", 400)
                    }
                }
                const type = ["POS", "NEU", "NEG"]
                // const type = ["Que", "Ans", "Conf", "Hes", "SelfDes"]
                for (let i = 0; i < data.length; i++) {
                    svg.datum(type[i])
                        .append("text")
                        .attr("x", legendpadding/4)
                        .attr("y", function (d) { return (y(i)+y.bandwidth()/2) })
                        .text( function (d) { return d })
                        .attr("text-anchor", "middle")
                        .attr("font-family", "sans-serif")
                        .attr("alignment-baseline", "middle")
                        .attr("pointer-events", "none")
                        .attr("font-weight", 600)
                    svg.datum(type[i])
                        .append("text")
                        .attr("x", function (d) { return (x(i)+x.bandwidth()/2) })
                        .attr("y", legendpaddingwidth/2)
                        // .attr("transform", "rotate(-90)")
                        .text( function (d) { return d })
                        .attr("text-anchor", "middle")
                        .attr("font-family", "sans-serif")
                        .attr("alignment-baseline", "middle")
                        .attr("pointer-events", "none")
                        .attr("font-weight", 600)
                }
            });
        }   
    }

};
</script>

<style>
.performanceView {
    /* background-color: lightgray; */
    height: 85%;
}

.performance {
    height: 100%;
}

.modelperformanceContainer {
    height: 20vh;
    width: 100%;
}

.el-table__row {
    font-size: 15px;
    font-family: sans-serif;
}
</style>
