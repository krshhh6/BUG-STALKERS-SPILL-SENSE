import React, { useMemo } from 'react';
import type { TabType } from '../types/dashboard';
import { LimelightNav, type NavItem } from '@/components/ui/limelight-nav';
import { Map, FlaskConical, Ship, ShieldCheck, Menu, X, Globe2 } from 'lucide-react';

interface MobileNavProps {
  activeTab: TabType;
  onSelectTab: (tab: TabType) => void;
  isDrawerOpen: boolean;
  onToggleDrawer: () => void;
  incidentCount?: number;
}

export const MobileNav: React.FC<MobileNavProps> = ({
  activeTab,
  onSelectTab,
  isDrawerOpen,
  onToggleDrawer,
  incidentCount = 4,
}) => {
  // Determine active index for Limelight spotlight:
  // 0: dashboard (Map)
  // 1: detection (SAR Lab)
  // 2: attribution (Vessels)
  // 3: evidence (Evidence)
  // 4: menu / drawer
  const activeIndex = useMemo(() => {
    if (isDrawerOpen) return 5;
    switch (activeTab) {
      case 'dashboard':
        return 0;
      case 'detection':
        return 1;
      case 'attribution':
        return 2;
      case 'evidence':
        return 3;
      case 'globe':
        return 4;
      default:
        return 5; // drift or analytics routes
    }
  }, [activeTab, isDrawerOpen]);

  const navItems: NavItem[] = useMemo(
    () => [
      {
        id: 'nav-map',
        icon: <Map className="w-5 h-5" />,
        label: 'Map',
        badge: incidentCount > 0 ? incidentCount : undefined,
        onClick: () => {
          if (isDrawerOpen) onToggleDrawer();
          onSelectTab('dashboard');
        },
      },
      {
        id: 'nav-detection',
        icon: <FlaskConical className="w-5 h-5" />,
        label: 'SAR Lab',
        onClick: () => {
          if (isDrawerOpen) onToggleDrawer();
          onSelectTab('detection');
        },
      },
      {
        id: 'nav-attribution',
        icon: <Ship className="w-5 h-5" />,
        label: 'Vessels',
        onClick: () => {
          if (isDrawerOpen) onToggleDrawer();
          onSelectTab('attribution');
        },
      },
      {
        id: 'nav-evidence',
        icon: <ShieldCheck className="w-5 h-5" />,
        label: 'Evidence',
        onClick: () => {
          if (isDrawerOpen) onToggleDrawer();
          onSelectTab('evidence');
        },
      },
      {
        id: 'nav-globe',
        icon: <Globe2 className="w-5 h-5" />,
        label: '3D Globe',
        onClick: () => {
          if (isDrawerOpen) onToggleDrawer();
          onSelectTab('globe');
        },
      },
      {
        id: 'nav-menu',
        icon: isDrawerOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />,
        label: isDrawerOpen ? 'Close' : 'Menu',
        onClick: onToggleDrawer,
      },
    ],
    [incidentCount, isDrawerOpen, onSelectTab, onToggleDrawer]
  );

  return (
    <nav className="mobile-bottom-nav" aria-label="Mobile Command Navigation">
      <LimelightNav
        items={navItems}
        activeIndex={activeIndex}
        className="w-full max-w-[420px] mx-auto h-[60px] bg-white/95 border-slate-200/90 shadow-[0_8px_30px_rgba(0,0,0,0.08)] backdrop-blur-md rounded-2xl justify-between px-2"
        iconContainerClassName="px-1"
      />
    </nav>
  );
};
