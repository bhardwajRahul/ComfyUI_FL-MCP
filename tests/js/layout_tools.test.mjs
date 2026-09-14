import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";


const root = new URL("../../", import.meta.url);


async function loadBrowserClass(relativePath, className, globals = {}) {
    let source = await readFile(new URL(relativePath, root), "utf8");
    source = source.replace(
        /^import\s+[\s\S]*?\s+from\s+["'][^"']+["'];\s*/gm,
        "",
    );
    source = source.replace(`export class ${className}`, `class ${className}`);
    source += `\nglobalThis.__loadedClass = ${className};\n`;
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {}, debug() {} },
        structuredClone,
        URLSearchParams,
        setTimeout,
        clearTimeout,
        ...globals,
    });
    vm.runInContext(source, context, { filename: relativePath });
    return context.__loadedClass;
}


function node(id, x, y, width = 100, height = 80, title = `Node ${id}`) {
    return {
        id,
        title,
        type: "TestNode",
        comfyClass: "TestNode",
        pos: [x, y],
        size: [width, height],
        inputs: [],
        outputs: [],
    };
}


function appHarness(nodes) {
    const workflow = {
        key: "workflow-a",
        changeTracker: {
            changeCount: 0,
        },
    };
    const graph = {
        _nodes: nodes,
        _groups: [],
        links: {},
        changeCalls: 0,
        serialize() {
            return {
                nodes: this._nodes.map(item => ({
                    id: item.id,
                    type: item.type,
                    pos: [...item.pos],
                    size: [...item.size],
                })),
            };
        },
        change() { this.changeCalls += 1; },
        setDirtyCanvas() {},
    };
    const canvas = {
        read_only: false,
        beforeCalls: 0,
        afterCalls: 0,
        emitBeforeChange() {
            this.beforeCalls += 1;
            workflow.changeTracker.changeCount += 1;
        },
        emitAfterChange() {
            this.afterCalls += 1;
            workflow.changeTracker.changeCount -= 1;
        },
        setDirty() {},
    };
    return {
        app: {
            graph,
            canvas,
            extensionManager: { workflow: { activeWorkflow: workflow } },
        },
        workflow,
        graph,
        canvas,
    };
}


async function loadFlApi(harness) {
    return await loadBrowserClass("web/js/fl_api.js", "FL_API", {
        app: harness.app,
        api: { dispatchCustomEvent() {} },
        nodeIdsEqual: (left, right) => String(left) === String(right),
        workflowGraphHash: async value => JSON.stringify(value),
        workflowGraphHashExcludingExtra: async value => JSON.stringify(value),
        canonicalWorkflowJSON: value => JSON.stringify(value),
        GRAPH_PRECONDITION_SCHEMA: "test",
    });
}


test("layout calculation is side-effect free, anchored, and refuses groups", async () => {
    const nodes = [node(1, 120, 240), node(2, 500, 420)];
    const harness = appHarness(nodes);
    const LayoutEngine = await loadBrowserClass(
        "web/js/layout_engine.js",
        "LayoutEngine",
        { app: harness.app },
    );
    const engine = new LayoutEngine();
    const before = nodes.map(item => [...item.pos]);

    const layout = engine.calculateLayout(["1", "2"], "grid");

    assert.deepEqual(nodes.map(item => [...item.pos]), before);
    assert.equal(Math.min(...layout.map(item => item.x)), 120);
    assert.equal(Math.min(...layout.map(item => item.y)), 240);
    harness.graph._groups.push({ title: "Grouped" });
    assert.throws(
        () => engine.calculateLayout(null, "flow_horizontal"),
        /cannot safely preserve workflow groups/,
    );
});


test("FL_API resolves layout IDs and applies one atomic change", async () => {
    const nodes = [node(1, 10, 20), node("two", 300, 200, 120, 90, "Second")];
    const harness = appHarness(nodes);
    const FL_API = await loadFlApi(harness);
    const flApi = new FL_API();
    flApi.layoutEngine = {
        setSpacingMultiplier(value) { assert.equal(value, 1.5); },
        calculateLayout(ids, strategy) {
            assert.deepEqual([...ids], [1, "two"]);
            assert.equal(strategy, "flow_vertical");
            return [
                { node_id: 1, x: 50, y: 60, width: 100, height: 80 },
                { node_id: "two", x: 50, y: 180, width: 120, height: 90 },
            ];
        },
    };

    const observed = flApi.getLayout(["1", "Second"]);
    assert.equal(observed.count, 2);
    assert.deepEqual([...observed.nodes.map(item => item.node_id)], [1, "two"]);

    const results = await flApi.modifyLayout(null, {
        auto_layout: true,
        node_ids: ["1", "Second"],
        strategy: "flow_vertical",
        spacing_multiplier: 1.5,
    });

    assert.deepEqual(nodes.map(item => [...item.pos]), [[50, 60], [50, 180]]);
    assert.equal(results.every(item => item.success), true);
    assert.equal(harness.graph.changeCalls, 1);
    assert.equal(harness.canvas.beforeCalls, 1);
    assert.equal(harness.canvas.afterCalls, 1);
});


test("FL_API rolls every rectangle back when verification fails", async () => {
    const first = node(1, 10, 20);
    const second = node(2, 30, 40);
    const secondPosition = second.pos;
    second.pos = new Proxy(secondPosition, {
        set(target, property, value) {
            target[property] = property === "0" && value === 300 ? 301 : value;
            return true;
        },
    });
    const harness = appHarness([first, second]);
    const FL_API = await loadFlApi(harness);
    const flApi = new FL_API();

    await assert.rejects(
        flApi.modifyLayout([
            { node_id: 1, x: 100, y: 200 },
            { node_id: 2, x: 300, y: 400 },
        ]),
        /Layout verification failed for node 2/,
    );

    assert.deepEqual([...first.pos], [10, 20]);
    assert.deepEqual([...second.pos], [30, 40]);
    assert.equal(harness.graph.changeCalls, 0);
    assert.equal(harness.canvas.beforeCalls, 1);
    assert.equal(harness.canvas.afterCalls, 1);
});


test("FL_API rolls layout back if the active workflow changes", async () => {
    const target = node(1, 10, 20);
    const harness = appHarness([target]);
    const FL_API = await loadFlApi(harness);
    const flApi = new FL_API();
    flApi.acceptWorkflowMutationGuard = async () => {
        harness.app.extensionManager.workflow.activeWorkflow = { key: "workflow-b" };
        throw new Error("workflow changed during layout");
    };

    await assert.rejects(
        flApi.modifyLayout([{ node_id: 1, x: 100, y: 200 }]),
        /workflow changed during layout/,
    );

    assert.deepEqual(target.pos, [10, 20]);
    assert.equal(harness.canvas.read_only, false);
    assert.equal(harness.workflow.changeTracker.changeCount, 0);
});


test("FL_API rolls layout back if the Comfy change transaction cannot close", async () => {
    const target = node(1, 10, 20);
    const harness = appHarness([target]);
    harness.canvas.emitAfterChange = function () {
        this.afterCalls += 1;
        throw new Error("change transaction failed");
    };
    const FL_API = await loadFlApi(harness);
    const flApi = new FL_API();

    await assert.rejects(
        flApi.modifyLayout([{ node_id: 1, x: 100, y: 200 }]),
        /change transaction failed/,
    );

    assert.deepEqual(target.pos, [10, 20]);
    assert.equal(harness.canvas.read_only, false);
});


test("layout tool handlers expose the direct response and revision 2 contracts", async () => {
    class FakeFLApi {
        getLayout() { return { nodes: [], count: 0 }; }
        setSessionId() {}
    }
    class FakeQueryExecutor {}
    const ToolExecutor = await loadBrowserClass(
        "web/js/tool_executor.js",
        "ToolExecutor",
        {
            FL_API: FakeFLApi,
            QueryExecutor: FakeQueryExecutor,
            performance: { now: () => 0 },
        },
    );
    const executor = new ToolExecutor({ sessionId: null });

    assert.deepEqual(
        await executor.toolHandlers.get_layout({ node_ids: null }),
        { nodes: [], count: 0 },
    );
    assert.equal(executor.getToolContractRevisions().get_layout, 2);
    assert.equal(executor.getToolContractRevisions().modify_layout, 2);
});
