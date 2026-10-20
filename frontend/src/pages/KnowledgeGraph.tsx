import React, { useEffect, useRef, useState } from 'react';
import { Card, Col, Drawer, Empty, Row, Spin, Statistic, Tag, Typography, message } from 'antd';
import * as d3 from 'd3';

const { Title, Text } = Typography;

// === Visual constants (4 entity types + 5 relation types found in real Neo4j) ===

const NODE_COLORS: Record<string, string> = {
  Clothing: '#1890ff',
  Style: '#fa8c16',
  Category: '#f5222d',
  Brand: '#52c41a',
};

const NODE_RADIUS: Record<string, number> = {
  Clothing: 6,
  Style: 16,
  Category: 16,
  Brand: 11,
};

const EDGE_COLORS: Record<string, string> = {
  HAS_STYLE: '#fa8c16',
  IN_CATEGORY: '#f5222d',
  FROM_BRAND: '#52c41a',
  MATCHES_WITH: '#722ed1',
  SIMILAR_TO: '#eb2f96',
};

const NODE_LABEL_KEYS: Record<string, string> = {
  Clothing: 'name',
  Style: 'name',
  Category: 'name',
  Brand: 'name',
};

const API = 'http://localhost:8000';

interface KGNode {
  id: string;
  neo4j_id: number;
  type: string;
  props: Record<string, unknown>;
}

interface KGEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

interface KGStats {
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  total_nodes: number;
  total_relationships: number;
}

const KnowledgeGraph: React.FC = () => {
  const [stats, setStats] = useState<KGStats | null>(null);
  const [graph, setGraph] = useState<{ nodes: KGNode[]; edges: KGEdge[] }>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<KGNode | null>(null);
  const [sampleSize, setSampleSize] = useState(80);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Fetch stats + sampled subgraph on mount
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/kg/stats`).then((r) => r.json()),
      fetch(`${API}/api/kg/graph?limit=${sampleSize}`).then((r) => r.json()),
    ])
      .then(([s, g]) => {
        if (cancelled) return;
        setStats(s);
        setGraph({ nodes: g.nodes ?? [], edges: g.edges ?? [] });
      })
      .catch((err) => {
        console.error(err);
        message.error('加载知识图谱失败');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sampleSize]);

  // Render D3 force-directed graph whenever graph data changes
  useEffect(() => {
    if (!graph.nodes.length || !svgRef.current) return;

    const svgEl = svgRef.current;
    const width = svgEl.clientWidth || 900;
    const height = 540;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    // Container group for zoom/pan
    const container = svg.append('g').attr('class', 'kg-container');

    // Arrow marker for directed edges
    svg
      .append('defs')
      .selectAll('marker')
      .data(Object.entries(EDGE_COLORS))
      .enter()
      .append('marker')
      .attr('id', (d) => `arrow-${d[0]}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', (d) => d[1]);

    // Clone nodes so d3 can mutate x/y/vx/vy
    const nodes = graph.nodes.map((n) => ({ ...n })) as Array<KGNode & { x?: number; y?: number; vx?: number; vy?: number }>;
    const links = graph.edges.map((e) => ({ ...e }));

    const sim = d3
      .forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force(
        'link',
        d3
          .forceLink(links as d3.SimulationLinkDatum<d3.SimulationNodeDatum>[])
          .id((d: any) => d.id)
          .distance(60)
          .strength(0.4),
      )
      .force('charge', d3.forceManyBody().strength(-180))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide<any>().radius((d) => (NODE_RADIUS[d.type] || 8) + 4))
      .force('x', d3.forceX(width / 2).strength(0.04))
      .force('y', d3.forceY(height / 2).strength(0.04));

    const link = container
      .append('g')
      .attr('class', 'kg-links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', (d) => EDGE_COLORS[d.type] || '#aaa')
      .attr('stroke-opacity', 0.55)
      .attr('stroke-width', (d) => Math.min(2.5, 0.6 + (d.weight || 1) * 0.4))
      .attr('marker-end', (d) => `url(#arrow-${d.type})`);

    const node = container
      .append('g')
      .attr('class', 'kg-nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'kg-node')
      .style('cursor', 'pointer')
      .on('click', (_event, d) => setSelected(d as KGNode))
      .call(
        d3
          .drag<SVGGElement, any>()
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            (d as any).fx = (d as any).x;
            (d as any).fy = (d as any).y;
          })
          .on('drag', (event, d) => {
            (d as any).fx = event.x;
            (d as any).fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            (d as any).fx = null;
            (d as any).fy = null;
          }),
      );

    node
      .append('circle')
      .attr('r', (d) => NODE_RADIUS[d.type] || 8)
      .attr('fill', (d) => NODE_COLORS[d.type] || '#888')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    node
      .append('text')
      .text((d) => {
        const key = NODE_LABEL_KEYS[d.type] || 'name';
        return String(d.props?.[key] ?? d.id).slice(0, 14);
      })
      .attr('font-size', 10)
      .attr('text-anchor', 'middle')
      .attr('dy', (d) => -(NODE_RADIUS[d.type] || 8) - 4)
      .attr('fill', '#333')
      .attr('pointer-events', 'none');

    sim.on('tick', () => {
      link
        .attr('x1', (d: any) => (d.source.x ?? 0))
        .attr('y1', (d: any) => (d.source.y ?? 0))
        .attr('x2', (d: any) => (d.target.x ?? 0))
        .attr('y2', (d: any) => (d.target.y ?? 0));
      node.attr('transform', (d: any) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    // Zoom / pan
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform.toString());
      });
    svg.call(zoom as any);

    return () => {
      sim.stop();
    };
  }, [graph]);

  return (
    <div style={{ padding: 24, background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={3} style={{ marginTop: 0 }}>
        知识图谱可视化
      </Title>
      <Text type="secondary">
        数据源: Neo4j 5(localhost:7687);本视图为类型配额采样,Style / Category 全量,Brand / Clothing 随机抽样。
      </Text>

      <Spin spinning={loading} tip="加载中...">
        {stats && (
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            {(['Clothing', 'Style', 'Category', 'Brand'] as const).map((label) => (
              <Col key={label} span={4}>
                <Card size="small" style={{ borderTop: `3px solid ${NODE_COLORS[label]}` }}>
                  <Statistic
                    title={label}
                    value={stats.node_counts[label] ?? 0}
                    valueStyle={{ color: NODE_COLORS[label], fontSize: 22 }}
                  />
                </Card>
              </Col>
            ))}
            <Col span={4}>
              <Card size="small" style={{ borderTop: '3px solid #722ed1' }}>
                <Statistic
                  title="关系总数"
                  value={stats.total_relationships}
                  valueStyle={{ color: '#722ed1', fontSize: 22 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ borderTop: '3px solid #13c2c2' }}>
                <Statistic
                  title="节点总数"
                  value={stats.total_nodes}
                  valueStyle={{ color: '#13c2c2', fontSize: 22 }}
                />
              </Card>
            </Col>
          </Row>
        )}

        <Card
          title="图谱结构(力导向布局)"
          style={{ marginTop: 16 }}
          extra={
            <div>
              <Text type="secondary" style={{ marginRight: 8 }}>
                采样大小:
              </Text>
              {[40, 80, 150].map((n) => (
                <Tag.CheckableTag
                  key={n}
                  checked={sampleSize === n}
                  onChange={() => setSampleSize(n)}
                  style={{ marginRight: 4 }}
                >
                  {n}
                </Tag.CheckableTag>
              ))}
              <Tag color="default" style={{ marginLeft: 8 }}>
                节点 {graph.nodes.length} · 边 {graph.edges.length}
              </Tag>
            </div>
          }
        >
          {graph.nodes.length === 0 ? (
            <Empty description="无可视化节点" />
          ) : (
            <svg
              ref={svgRef}
              width="100%"
              height={540}
              style={{
                background: 'linear-gradient(135deg, #fafafa 0%, #f0f2f5 100%)',
                borderRadius: 8,
                cursor: 'grab',
              }}
            />
          )}
        </Card>

        {stats && (
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={12}>
              <Card title="实体类型" size="small">
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                  {Object.entries(NODE_COLORS).map(([label, color]) => (
                    <Tag key={label} color={color} style={{ padding: '4px 12px', fontSize: 13 }}>
                      {label} · {stats.node_counts[label] ?? 0}
                    </Tag>
                  ))}
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="关系类型" size="small">
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                  {Object.entries(stats.relationship_counts).map(([rel, cnt]) => (
                    <Tag
                      key={rel}
                      color={EDGE_COLORS[rel] ? undefined : 'default'}
                      style={{
                        background: EDGE_COLORS[rel] || undefined,
                        borderColor: EDGE_COLORS[rel] || undefined,
                        color: EDGE_COLORS[rel] ? '#fff' : undefined,
                        padding: '4px 12px',
                        fontSize: 13,
                      }}
                    >
                      {rel} · {cnt}
                    </Tag>
                  ))}
                </div>
              </Card>
            </Col>
          </Row>
        )}
      </Spin>

      <Drawer
        title={
          selected ? (
            <span>
              <Tag color={NODE_COLORS[selected.type] || '#888'}>{selected.type}</Tag>
              节点详情
            </span>
          ) : (
            '节点详情'
          )
        }
        open={!!selected}
        onClose={() => setSelected(null)}
        width={420}
      >
        {selected && (
          <div>
            <Title level={5} style={{ marginTop: 0 }}>
              {String(selected.props?.[NODE_LABEL_KEYS[selected.type] || 'name'] ?? selected.id)}
            </Title>
            <div style={{ marginBottom: 12, color: '#888' }}>
              Neo4j id: <code>{selected.neo4j_id}</code>
            </div>
            <Title level={5}>属性</Title>
            <pre
              style={{
                background: '#f5f5f5',
                padding: 12,
                borderRadius: 6,
                fontSize: 12,
                maxHeight: 400,
                overflow: 'auto',
              }}
            >
              {JSON.stringify(selected.props, null, 2)}
            </pre>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default KnowledgeGraph;
