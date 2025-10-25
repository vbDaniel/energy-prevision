import { Home, TrendingUp, Bell, Settings, Zap, BarChart3 } from "lucide-react";
import {
  SidebarWrap,
  Inner,
  Header,
  LogoBox,
  TitleWrap,
  Title,
  SubTitle,
  Nav,
  MenuButton,
  StatsContainer,
  StatsCard,
  StatTitle,
  StatValue,
  StatChange,
} from "./styles";

const menuItems = [
  { icon: Home, label: "Dashboard", active: true },
  { icon: TrendingUp, label: "Previsões", active: false },
  { icon: BarChart3, label: "Relatórios", active: false },
  { icon: Bell, label: "Alertas", active: false },
  { icon: Settings, label: "Configurações", active: false },
];

export const Sidebar = () => {
  return (
    <SidebarWrap>
      <Inner>
        <Header>
          <LogoBox>
            <Zap />
          </LogoBox>
          <TitleWrap>
            <Title>EnergyFlow</Title>
            <SubTitle>Gestão Inteligente</SubTitle>
          </TitleWrap>
        </Header>

        <Nav>
          {menuItems.map((item) => (
            <MenuButton key={item.label} $active={item.active}>
              <item.icon />
              <span>{item.label}</span>
            </MenuButton>
          ))}
        </Nav>
      </Inner>

      <StatsContainer>
        <StatsCard>
          <StatTitle>Consumo do Mês</StatTitle>
          <StatValue>342 kWh</StatValue>
          <StatChange>-12% vs mês anterior</StatChange>
        </StatsCard>
      </StatsContainer>
    </SidebarWrap>
  );
};
