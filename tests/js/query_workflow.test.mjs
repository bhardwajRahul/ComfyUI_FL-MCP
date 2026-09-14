import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";


const root = new URL("../../", import.meta.url);


async function loadQueryExecutor(nodes) {
    const source = await readFile(new URL("web/js/query_executor.js", root), "utf8");
    const transformed = source
        .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
        .replace(/^import[\s\S]*?;$/gm, "")
        .replace("export class QueryExecutor", "class QueryExecutor");
    const context = vm.createContext({ app: { graph: { _nodes: nodes } }, console });
    vm.runInContext(`${transformed}\nglobalThis.QueryExecutor = QueryExecutor;`, context);
    return new context.QueryExecutor();
}


function workflowNodes(count) {
    return Array.from({ length: count }, (_, index) => ({
        id: index + 1,
        type: index % 2 === 0 ? "KSampler" : "PreviewImage",
        comfyClass: index % 2 === 0 ? "KSampler" : "PreviewImage",
        title: `Node ${index + 1}`,
        pos: [index * 20, index * 10],
        size: [180, 100],
        mode: 0,
        widgets: [],
        inputs: [],
        outputs: [],
    }));
}


test("workflow queries return bounded pagination metadata", async () => {
    const executor = await loadQueryExecutor(workflowNodes(12));
    const result = executor.execute({ result_format: "summary", offset: 3, limit: 4 });

    assert.equal(result.count, 4);
    assert.equal(result.total, 12);
    assert.equal(result.offset, 3);
    assert.equal(result.limit, 4);
    assert.equal(result.has_more, true);
    assert.equal(result.next_offset, 7);
    assert.equal(result.results[0].id, 4);
    assert.deepEqual(Object.keys(result.results[0]), ["id", "type", "title"]);
});


test("full workflow queries include positions and connections only when requested", async () => {
    const executor = await loadQueryExecutor(workflowNodes(2));
    const compact = executor.execute({ result_format: "full", limit: 1 });
    const detailed = executor.execute({
        result_format: "full",
        limit: 1,
        include_position: true,
        include_connections: true,
    });

    assert.equal("position" in compact.results[0], false);
    assert.equal("connections" in compact.results[0], false);
    assert.equal("position" in detailed.results[0], true);
    assert.equal("connections" in detailed.results[0], true);
});


test("a direct filter condition works without a logical wrapper", async () => {
    const executor = await loadQueryExecutor(workflowNodes(6));
    const direct = executor.execute({
        filters: { field: "type", operator: "equals", value: "KSampler" },
        result_format: "summary",
    });
    const wrapped = executor.execute({
        filters: {
            operator: "and",
            filters: [{ field: "type", operator: "equals", value: "KSampler" }],
        },
        result_format: "summary",
    });

    assert.equal(direct.count, 3);
    assert.deepEqual(direct, wrapped);
});
