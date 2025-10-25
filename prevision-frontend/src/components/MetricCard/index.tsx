import { LucideIcon } from "lucide-react";
import CardModule from "@/components/ui/Card";
import {
  MetricCardWrap,
  Row,
  Left,
  SmallTitle,
  Value,
  Change,
  IconBox,
} from "./styles";

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  icon: LucideIcon;
  trend: "up" | "down" | "neutral";
  iconBgClass?: string; // mantido para compatibilidade, não usado diretamente
}

export const MetricCard = ({
  title,
  value,
  change,
  icon: Icon,
  trend,
  iconBgClass = "bg-primary/10",
}: MetricCardProps) => {
  return (
    <MetricCardWrap as={CardModule.Card}>
      <Row>
        <Left>
          <SmallTitle>{title}</SmallTitle>
          <Value>{value}</Value>
          <Change $trend={trend}>{change}</Change>
        </Left>
        <IconBox>
          <Icon size={24} color={"hsl(var(--primary))"} />
        </IconBox>
      </Row>
    </MetricCardWrap>
  );
};
