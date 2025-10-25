import styled, { css } from "styled-components";

export type AlertVariant = "default" | "destructive";

export const StyledAlert = styled.div<{ variant?: AlertVariant }>`
  position: relative; /* relative */
  width: 100%; /* w-full */
  border-radius: var(--radius); /* rounded-lg */
  border: 1px solid hsl(var(--border)); /* border */
  padding: 1rem; /* p-4 */
  background-color: hsl(var(--background));
  color: hsl(var(--foreground));

  /* Icon placement and sibling offsets */
  & > svg {
    position: absolute; /* [&>svg]:absolute */
    left: 1rem; /* [&>svg]:left-4 */
    top: 1rem; /* [&>svg]:top-4 */
    color: hsl(var(--foreground)); /* [&>svg]:text-foreground */
  }

  /* Apply padding-left to all siblings after the SVG */
  & > svg ~ * {
    padding-left: 1.75rem; /* [&>svg~*]:pl-7 */
  }

  /* Slight vertical adjustment for the immediate next div after the SVG */
  & > svg + div {
    transform: translateY(-3px); /* [&>svg+div]:translate-y-[-3px] */
  }

  ${({ variant }) =>
    variant === "destructive"
      ? css`
          border-color: hsl(var(--destructive)); /* dark:border-destructive */
          color: hsl(var(--destructive)); /* text-destructive */
          & > svg {
            color: hsl(var(--destructive)); /* [&>svg]:text-destructive */
          }
        `
      : css``}
`;

export const StyledAlertTitle = styled.h5`
  margin-bottom: 0.25rem; /* mb-1 */
  font-weight: 500; /* font-medium */
  line-height: 1; /* leading-none */
  letter-spacing: -0.025em; /* tracking-tight */
`;

export const StyledAlertDescription = styled.div`
  font-size: 0.875rem; /* text-sm */
  line-height: 1.25rem;

  /* Ensure paragraphs inside have relaxed leading */
  & p {
    line-height: 1.625; /* [&_p]:leading-relaxed */
  }
`;
