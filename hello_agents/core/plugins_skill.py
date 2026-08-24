"""SkillPlugin - 技能系统插件

职责：SkillLoader 初始化、SkillTool 自动注册
"""

from typing import Optional
from pathlib import Path
from ..skills.loader import SkillLoader
from ..tools.builtin.skill_tool import SkillTool
from .plugins import AgentPlugin, PluginContext


class SkillPlugin(AgentPlugin):
    """技能系统插件"""
    
    name = "skill"
    priority = 70
    
    def __init__(self, config=None):
        super().__init__(config)
        self._skill_loader: Optional[SkillLoader] = None
    
    def _initialize(self) -> None:
        if not self.config.skills_enabled:
            return
        
        skills_path = Path(self.config.skills_dir)
        self._skill_loader = SkillLoader(skills_dir=skills_path)
        self.context.skill_loader = self._skill_loader
        
        # 自动注册 SkillTool
        if self.config.skills_auto_register and self.context.tool_registry:
            skill_tool = SkillTool(skill_loader=self._skill_loader)
            self.context.tool_registry.register_tool(skill_tool)
    
    @property
    def skill_loader(self) -> Optional[SkillLoader]:
        return self._skill_loader