import { useConsole } from '../context/ConsoleContext';

// Every number on the console is clickable — opens the Evidence Drawer
// showing the field it came from, the equation, and the live value.
export default function Field({ label, field, equation, value, className = '', children, block = false }) {
  const { openDrawer } = useConsole();
  const Tag = block ? 'div' : 'span';
  return (
    <Tag
      className={`field-clickable ${className}`}
      role="button"
      tabIndex={0}
      onClick={() => openDrawer(field ?? label, equation, value)}
      onKeyDown={(e) => { if (e.key === 'Enter') openDrawer(field ?? label, equation, value); }}
      title={`inspect ${field ?? label}`}
    >
      {children}
    </Tag>
  );
}
