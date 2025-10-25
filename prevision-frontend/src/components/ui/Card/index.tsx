import React from "react";
import {
  StyledCard,
  StyledCardHeader,
  StyledCardTitle,
  StyledCardDescription,
  StyledCardContent,
  StyledCardFooter,
} from "./styles";

export const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ children, ...props }, ref) => (
  <StyledCard ref={ref} {...props}>
    {children}
  </StyledCard>
));
Card.displayName = "Card";

export const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ children, ...props }, ref) => (
  <StyledCardHeader ref={ref} {...props}>
    {children}
  </StyledCardHeader>
));
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ children, ...props }, ref) => (
  <StyledCardTitle ref={ref} {...props}>
    {children}
  </StyledCardTitle>
));
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ children, ...props }, ref) => (
  <StyledCardDescription ref={ref} {...props}>
    {children}
  </StyledCardDescription>
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ children, ...props }, ref) => (
  <StyledCardContent ref={ref} {...props}>
    {children}
  </StyledCardContent>
));
CardContent.displayName = "CardContent";

export const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ children, ...props }, ref) => (
  <StyledCardFooter ref={ref} {...props}>
    {children}
  </StyledCardFooter>
));
CardFooter.displayName = "CardFooter";

export default {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
};
