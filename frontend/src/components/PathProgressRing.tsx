import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface PathProgressRingProps {
  completed: number;
  total: number;
  size?: number;
}

export const PathProgressRing = ({
  completed,
  total,
  size = 120
}: PathProgressRingProps) => {
  const percentage = (completed / total) * 100;

  const data = [
    { name: 'Completed', value: completed, fill: '#10B981' }, // Green
    { name: 'Remaining', value: total - completed, fill: '#E5E7EB' } // Gray
  ];

  // Custom label showing percentage
  const renderLabel = ({ cx, cy }: { cx: number; cy: number }) => {
    return (
      <text
        x={cx}
        y={cy}
        fill="#1F2937"
        textAnchor="middle"
        dominantBaseline="central"
        className="font-bold text-2xl"
      >
        {Math.round(percentage)}%
      </text>
    );
  };

  return (
    <ResponsiveContainer width={size} height={size}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={size * 0.6} // Creates the ring effect
          outerRadius={size * 0.8}
          startAngle={90}
          endAngle={-270} // Clockwise from top
          dataKey="value"
          label={renderLabel}
          labelLine={false}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Pie>
      </PieChart>
    </ResponsiveContainer>
  );
};
