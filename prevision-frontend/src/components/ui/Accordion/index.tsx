import * as React from "react";
import { ChevronDown } from "lucide-react";
import {
  AccordionWrap,
  AccordionItemWrap,
  TriggerButton,
  Chevron as ChevronWrap,
  ContentWrap,
  ContentInner,
} from "./styles";

// Types
type AccordionType = "single" | "multiple";

// Context to manage Accordion state
interface AccordionContextValue {
  type: AccordionType;
  collapsible: boolean;
  openValues: string[];
  toggle: (val: string) => void;
}

const AccordionContext = React.createContext<AccordionContextValue | null>(
  null
);

export interface AccordionProps {
  children: React.ReactNode;
  type?: AccordionType;
  collapsible?: boolean;
  defaultValue?: string | string[];
  value?: string | string[];
  onValueChange?: (val: string | string[]) => void;
  className?: string;
}

export const Accordion: React.FC<AccordionProps> = ({
  children,
  type = "single",
  collapsible = true,
  defaultValue,
  value,
  onValueChange,
  className,
}) => {
  const controlled = value !== undefined;

  const [internal, setInternal] = React.useState<string[]>(() => {
    if (value !== undefined) {
      return Array.isArray(value) ? value : value ? [value] : [];
    }
    if (defaultValue !== undefined) {
      return Array.isArray(defaultValue)
        ? defaultValue
        : defaultValue
        ? [defaultValue]
        : [];
    }
    return [];
  });

  const openValues = controlled
    ? Array.isArray(value)
      ? value
      : value
      ? [value]
      : []
    : internal;

  const setOpenValues = (next: string[]) => {
    if (controlled) {
      onValueChange?.(type === "single" ? next[0] ?? "" : next);
    } else {
      setInternal(next);
    }
  };

  const toggle = (val: string) => {
    const isOpen = openValues.includes(val);
    if (type === "single") {
      if (isOpen) {
        setOpenValues(collapsible ? [] : openValues);
      } else {
        setOpenValues([val]);
      }
    } else {
      if (isOpen) {
        setOpenValues(openValues.filter((v) => v !== val));
      } else {
        setOpenValues([...openValues, val]);
      }
    }
  };

  return (
    <AccordionContext.Provider
      value={{ type, collapsible, openValues, toggle }}
    >
      <AccordionWrap className={className}>{children}</AccordionWrap>
    </AccordionContext.Provider>
  );
};

// Item context
interface ItemContextValue {
  value: string;
  open: boolean;
}
const ItemContext = React.createContext<ItemContextValue | null>(null);

export interface AccordionItemProps
  extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

export const AccordionItem = React.forwardRef<
  HTMLDivElement,
  AccordionItemProps
>(({ value, children, className, ...props }, ref) => {
  const ctx = React.useContext(AccordionContext);
  const open = !!ctx?.openValues.includes(value);

  return (
    <ItemContext.Provider value={{ value, open }}>
      <AccordionItemWrap ref={ref} className={className} {...props}>
        {children}
      </AccordionItemWrap>
    </ItemContext.Provider>
  );
});
AccordionItem.displayName = "AccordionItem";

export interface AccordionTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {}

export const AccordionTrigger = React.forwardRef<
  HTMLButtonElement,
  AccordionTriggerProps
>(({ children, className, ...props }, ref) => {
  const acc = React.useContext(AccordionContext);
  const item = React.useContext(ItemContext);

  if (!acc || !item) return null;

  const onClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    props.onClick?.(e);
    acc.toggle(item.value);
  };

  return (
    <TriggerButton
      ref={ref}
      aria-expanded={item.open}
      aria-controls={`accordion-content-${item.value}`}
      data-state={item.open ? "open" : "closed"}
      onClick={onClick}
      className={className}
      {...props}
    >
      <span style={{ flex: 1 }}>{children}</span>
      <ChevronWrap $open={item.open}>
        <ChevronDown size={16} />
      </ChevronWrap>
    </TriggerButton>
  );
});
AccordionTrigger.displayName = "AccordionTrigger";

export interface AccordionContentProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const AccordionContent = React.forwardRef<
  HTMLDivElement,
  AccordionContentProps
>(({ children, className, ...props }, ref) => {
  const item = React.useContext(ItemContext);
  const innerRef = React.useRef<HTMLDivElement>(null);
  const [height, setHeight] = React.useState(0);

  React.useLayoutEffect(() => {
    if (innerRef.current) {
      setHeight(innerRef.current.scrollHeight);
    }
  }, [children, item?.open]);

  if (!item) return null;

  return (
    <ContentWrap
      id={`accordion-content-${item.value}`}
      role="region"
      aria-labelledby={`accordion-trigger-${item.value}`}
      $open={item.open}
      $height={height}
      className={className}
      ref={ref}
      {...props}
    >
      <ContentInner ref={innerRef}>{children}</ContentInner>
    </ContentWrap>
  );
});
AccordionContent.displayName = "AccordionContent";

export { AccordionWrap };
