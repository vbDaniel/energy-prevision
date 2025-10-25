import { AlertTriangle, TrendingUp, Lightbulb } from "lucide-react";
import CardModule from "@/components/ui/Card";
import BadgeModule from "@/components/ui/Badge";
import {
  StyledCard,
  Title,
  AlertsList,
  AlertItem,
  IconWrap,
  Content,
  Row,
  AlertTitle,
  Message,
  BadgeWrapper,
} from "./styles";

const { Card } = CardModule;
const { Badge } = BadgeModule;
import type { ElementType } from "react";

type Alert = {
  type: "warning" | "info" | "tip";
  title: string;
  message: string;
  icon: ElementType;
  variant: "secondary" | "primary" | "muted";
};

const alerts: Alert[] = [
  {
    type: "warning",
    title: "Consumo acima do previsto",
    message: "Você está 8% acima da meta deste mês",
    icon: AlertTriangle,
    variant: "secondary",
  },
  {
    type: "info",
    title: "Pico de consumo detectado",
    message: "Consumo elevado entre 18h-20h",
    icon: TrendingUp,
    variant: "primary",
  },
  {
    type: "tip",
    title: "Dica de economia",
    message: "Desligue aparelhos em standby para economizar até 15%",
    icon: Lightbulb,
    variant: "muted",
  },
];

export const AlertCard = () => {
  return (
    <StyledCard as={Card}>
      <Title>Alertas e Notificações</Title>
      <AlertsList>
        {alerts.map((alert, index) => (
          <AlertItem key={index}>
            <IconWrap $variant={alert.variant}>
              <alert.icon />
            </IconWrap>
            <Content>
              <Row>
                <AlertTitle>{alert.title}</AlertTitle>
                {alert.type === "warning" && (
                  <BadgeWrapper>
                    <Badge variant="secondary">Importante</Badge>
                  </BadgeWrapper>
                )}
              </Row>
              <Message>{alert.message}</Message>
            </Content>
          </AlertItem>
        ))}
      </AlertsList>
    </StyledCard>
  );
};
